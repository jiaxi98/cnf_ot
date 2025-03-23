from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
from jaxtyping import Array

from cnf_ot.types import PRNGKey


def kl_loss_fn(
  model, dim: int, T: float, params: hk.Params, cond: float, rng: PRNGKey,
  batch_size: int
) -> Array:
  """KL-divergence loss function.
  KL-divergence between the normalizing flow and the reference
  distribution.

  NOTE: this loss function is used to fit multimodal distributions such as
  Gaussian mixture
  """

  def sample_source_fn(
    seed: PRNGKey,
    sample_shape: int,
  ):

    # gaussian source distribution
    A = jnp.array([[5, 1], [1, 0.5]])
    B = jnp.linalg.cholesky(A)
    return jax.random.normal(seed, shape=(sample_shape, dim)) @ B +\
      jnp.ones(dim).reshape(1, dim) * -3

    # gaussian mixture source distribution
    R = 5
    component_indices = jax.random.choice(
      seed, a=8, shape=(sample_shape, ), p=jnp.ones(8) / 8
    )
    sample_ = jnp.zeros((8, sample_shape, dim))
    sample_ = sample_.at[0].set(
      jax.random.normal(seed, shape=(sample_shape, dim)) + jnp.array([0.0, R])
    )
    sample_ = sample_.at[1].set(
      jax.random.normal(seed, shape=(sample_shape, dim)) + jnp.array([R, 0.0])
    )
    sample_ = sample_.at[2].set(
      jax.random.normal(seed, shape=(sample_shape, dim)) + jnp.array([0.0, -R])
    )
    sample_ = sample_.at[3].set(
      jax.random.normal(seed, shape=(sample_shape, dim)) + jnp.array([-R, 0.0])
    )
    sample_ = sample_.at[4].set(
      jax.random.normal(seed, shape=(sample_shape, dim)) +
      jnp.array([0.6 * R, 0.8 * R])
    )
    sample_ = sample_.at[5].set(
      jax.random.normal(seed, shape=(sample_shape, dim)) +
      jnp.array([0.6 * R, -0.8 * R])
    )
    sample_ = sample_.at[6].set(
      jax.random.normal(seed, shape=(sample_shape, dim)) +
      jnp.array([-0.6 * R, -0.8 * R])
    )
    sample_ = sample_.at[7].set(
      jax.random.normal(seed, shape=(sample_shape, dim)) +
      jnp.array([-0.6 * R, 0.8 * R])
    )

    sample = sample_[component_indices[jnp.arange(sample_shape)],
                    jnp.arange(sample_shape)]
    return sample

  def sample_target_fn(
    seed: PRNGKey,
    sample_shape: int,
  ):

    return jax.random.normal(seed, shape=(sample_shape, dim)) # +\
      # jnp.ones(dim).reshape(1, dim) * 3

  samples1 = sample_source_fn(seed=rng, sample_shape=batch_size)
  samples2 = sample_target_fn(seed=rng, sample_shape=batch_size)
  samples = samples1 * (T - cond) / T + samples2 * cond / T
  fake_cond_ = jnp.ones((1, )) * cond
  log_prob = model.apply.log_prob(params, samples, cond=fake_cond_)
  return -log_prob.mean()


# This function is used for debug. Try to see if the loss function of the
# simple ot problem is decreasing over optimization.
def ot_reverse_kl_loss_fn(
  model, dim: int, T: float, params: hk.Params, rng: PRNGKey,
  batch_size: int
) -> Array:
  
  target_prob1 = jax.vmap(
    partial(
      jax.scipy.stats.multivariate_normal.pdf,
      mean=jnp.ones(dim) * 3,
      cov=jnp.eye(dim)
    )
  )
  target_prob2 = jax.vmap(
    partial(
      jax.scipy.stats.multivariate_normal.pdf,
      mean=jnp.zeros(dim),
      cov=jnp.eye(dim)
    )
  )

  fake_cond_ = jnp.ones((batch_size, 1)) * 0
  samples, log_prob = model.apply.sample_and_log_prob(
    params,
    cond=fake_cond_,
    seed=rng,
    sample_shape=(batch_size, ),
  )
  loss = (log_prob - jnp.log(target_prob1(samples))).mean()
  fake_cond_ = jnp.ones((batch_size, 1))
  samples, log_prob = model.apply.sample_and_log_prob(
    params,
    cond=fake_cond_,
    seed=rng,
    sample_shape=(batch_size, ),
  )
  loss += (log_prob - jnp.log(target_prob2(samples))).mean()
  return loss
  

def reverse_kl_loss_fn(
  model, dim: int, T: float, beta: float, params: hk.Params, cond: float,
  rng: PRNGKey, batch_size: int
) -> Array:
  """Reverse KL-divergence loss function.
  Reverse KL-divergence between the normalizing flow and the reference distribution.
  """

  source_prob = jax.vmap(
    partial(
      jax.scipy.stats.multivariate_normal.pdf,
      mean=jnp.zeros(dim),
      cov=jnp.eye(dim) * 2 / beta * (T + 1)
    )
  )
  target_prob = jax.vmap(
    partial(
      jax.scipy.stats.multivariate_normal.pdf,
      mean=jnp.zeros(dim),
      cov=jnp.eye(dim) * 2 / beta,
    )
  )

  fake_cond_ = jnp.ones((batch_size, 1)) * cond
  samples, log_prob = model.apply.sample_and_log_prob(
    params,
    cond=fake_cond_,
    seed=rng,
    sample_shape=(batch_size, ),
  )
  return (
    log_prob - jnp.log(
      source_prob(samples) * (T - cond) / T + target_prob(samples) * cond / T
    )
  ).mean()


def density_fit_kl_loss_fn(
  model, dim: int, T: float, params: hk.Params, rng: PRNGKey, batch_size: int
) -> Array:

  return partial(kl_loss_fn, model, dim, T)(
    params, 0, rng, batch_size
  ) +\
    partial(kl_loss_fn, model, dim, T)(params, T, rng, batch_size)


def potential_loss_fn(
  model, dim: int, a: float, subtype: str, params: hk.Params, cond: float,
  rng: PRNGKey, batch_size: int
) -> Array:

  def quadratic_potential_fn(r: jnp.ndarray, ) -> jnp.ndarray:
    return jnp.sum(r**2, axis=1) / 2

  def double_well_potential_fn(r: jnp.ndarray, ) -> jnp.ndarray:
    return (
      jnp.linalg.norm(r - a * jnp.ones(dim).reshape(1, -1), axis=1) *
      jnp.linalg.norm(r + a * jnp.ones(dim).reshape(1, -1), axis=1) / 2
    )**2

  def obstacle_potential_fn(r: jnp.ndarray, ) -> jnp.ndarray:
    return 50 * jnp.exp(-jnp.sum(r**2, axis=1) / 2)

  fake_cond_ = jnp.ones((batch_size, 1)) * cond
  samples, _ = model.apply.sample_and_log_prob(
    params,
    cond=fake_cond_,
    seed=rng,
    sample_shape=(batch_size, ),
  )
  if subtype == "quadratic":
    return quadratic_potential_fn(samples).mean()
  elif subtype == "double_well":
    return double_well_potential_fn(samples).mean()
  elif subtype == "obstacle":
    return obstacle_potential_fn(samples).mean()


# def kinetic_loss_fn(t: float, params: hk.Params, rng: PRNGKey, batch_size: int) -> Array:
#     """Kinetic energy along the trajectory at time t
#     """
#     fake_cond_ = np.ones((batch_size, 1)) * t
#     samples = sample_fn(params, seed=rng, sample_shape=(batch_size, ), cond=fake_cond_)
#     xi = inverse_fn(params, samples, fake_cond_)
#     velocity = jax.jacfwd(partial(forward_fn, params, xi))(fake_cond_)
#     # velocity.shape = [batch_size, 2, batch_size, 1]
#     # velocity[jnp.arange(batch_size),:,jnp.arange(batch_size),0].shape = [batch_size, 2]
#     return jnp.mean(velocity[jnp.arange(batch_size),:,jnp.arange(batch_size),0]**2) * dim / 2


def kinetic_loss_fn(
  model, dim: int, dt: float, params: hk.Params, cond: float, rng: PRNGKey,
  batch_size: int
) -> Array:
  """Kinetic energy along the trajectory at time t
    """
  fake_cond_ = jnp.ones((batch_size, 1)) * (cond - dt / 2)
  # TODO: work out why this one does not work
  # r1 = sample_fn(
  #   params, seed=rng, sample_shape=(batch_size, ), cond=fake_cond_
  # )
  # xi = inverse_fn(params, r1, jnp.ones(1) * (t - dt / 2))
  # r2 = forward_fn(params, xi, jnp.ones(1) * (t - dt / 2))
  r1 = model.apply.sample(
    params, seed=rng, sample_shape=(batch_size, ), cond=fake_cond_
  )
  fake_cond_ = jnp.ones((batch_size, 1)) * (cond + dt / 2)
  r2 = model.apply.sample(
    params, seed=rng, sample_shape=(batch_size, ), cond=fake_cond_
  )
  velocity = (r2 - r1) / dt  # velocity.shape = [batch_size, 2]

  return jnp.mean(velocity**2) * dim / 2


def kinetic_with_score_loss_fn(
  model, dim: int, beta: float, dt: float, dx: float, params: hk.Params,
  cond: float, rng: PRNGKey, batch_size: int
) -> Array:
  """Kinetic energy along the trajectory at time t, notice that this contains
  not only the velocity but also the score function
  """
  fake_cond_ = jnp.ones((batch_size, 1)) * (cond - dt / 2)
  r1 = model.apply.sample(
    params, seed=rng, sample_shape=(batch_size, ), cond=fake_cond_
  )
  fake_cond_ = jnp.ones((batch_size, 1)) * (cond + dt / 2)
  r2 = model.apply.sample(
    params, seed=rng, sample_shape=(batch_size, ), cond=fake_cond_
  )
  fake_cond_ = jnp.ones((batch_size, 1)) * cond
  r3 = model.apply.sample(
    params, seed=rng, sample_shape=(batch_size, ), cond=fake_cond_
  )
  velocity = (r2 - r1) / dt
  # velocity += jax.jacfwd(partial(log_prob_fn, params, cond=jnp.ones(1) * t))(
  #   r3)[jnp.arange(batch_size),jnp.arange(batch_size)]/beta
  # finite difference approximation of the score function
  score = jnp.zeros((batch_size, dim))
  for i in range(dim):
    dr = jnp.zeros((1, dim))
    dr = dr.at[0, i].set(dx / 2)
    log_p1 = model.apply.log_prob(params, r3 + dr, cond=jnp.ones(1) * cond)
    log_p2 = model.apply.log_prob(params, r3 - dr, cond=jnp.ones(1) * cond)
    score = score.at[:, i].set((log_p1 - log_p2) / dx)
  velocity += score / beta
  return jnp.mean(velocity**2) * dim / 2


def flow_matching_loss_fn(
  model, dim: int, a: float, sigma: float, subtype: str, dt: float, dx: float,
  params: hk.Params, cond: float, rng: PRNGKey, batch_size: int
) -> Array:
    """Kinetic energy along the trajectory at time t, notice that this contains
    not only the velocity but also the score function
    """
    dt = 0.01
    fake_cond_ = jnp.ones((batch_size, 1)) * (cond - dt / 2)
    r1 = model.apply.sample(
      params, seed=rng, sample_shape=(batch_size, ), cond=fake_cond_
    )
    fake_cond_ = jnp.ones((batch_size, 1)) * (cond + dt / 2)
    r2 = model.apply.sample(
      params, seed=rng, sample_shape=(batch_size, ), cond=fake_cond_
    )
    fake_cond_ = jnp.ones((batch_size, 1)) * cond
    r3 = model.apply.sample(
      params, seed=rng, sample_shape=(batch_size, ), cond=fake_cond_
    )
    velocity = (r2 - r1) / dt
    score = jnp.zeros((batch_size, dim))
    dx = 0.01
    for i in range(dim):
      dr = jnp.zeros((1, dim))
      dr = dr.at[0, i].set(dx / 2)
      log_p1 = model.apply.log_prob(params, r3 + dr, cond=jnp.ones(1) * cond)
      log_p2 = model.apply.log_prob(params, r3 - dr, cond=jnp.ones(1) * cond)
      score = score.at[:, i].set((log_p1 - log_p2) / dx)
    velocity += score * sigma
    if subtype == "gradient":
      truth = -r3 * a
      x = r3[:, 0]
      y = r3[:, 1]
      # r1 = 4
      # r2 = 2
      # _pi = jnp.exp(-r1/4 * ((x - 6/5)**2 + (y - 6/5)**2 - .5)**2 -
      #                r2/2 * (y - 2)**2) +\
      #       jnp.exp(-r1/4 * ((x + 6/5)**2 + (y - 6/5)**2 - .5)**2 -
      #                r2/2 * (y - 2)**2) +\
      #       jnp.exp(-r1/4 * (x**2 + y**2 - 2)**2 - r2/2 * (y + 1)**2) + .01
      # dpidx = jnp.exp(-r1/4 * ((x - 6/5)**2 + (y - 6/5)**2 - .5)**2 -
      #                r2/2 * (y - 2)**2) * ((x - 6/5)**2 + (y - 6/5)**2 - .5) *\
      #                r1 * (x - 6/5) +\
      #         jnp.exp(-r1/4 * ((x + 6/5)**2 + (y - 6/5)**2 - .5)**2 -
      #                 r2/2 * (y - 2)**2) * ((x + 6/5)**2 + (y - 6/5)**2 - .5) *\
      #                k1 * (x + 6/5) +\
      #         jnp.exp(-r1/4 * (x**2 + y**2 - 2)**2 - r2/2 * (y + 1)**2) *\
      #                 (x**2 + y**2 - 2) * r1 * x
      # dpidy = jnp.exp(-r1/4 * ((x - 6/5)**2 + (y - 6/5)**2 - .5)**2 -
      #                r2/2 * (y - 2)**2) * (((x - 6/5)**2 + (y - 6/5)**2 - .5) *
      #                r1 * (y - 6/5) + r2 * (y - 2)) +\
      #         jnp.exp(-r1/4 * ((x + 6/5)**2 + (y - 6/5)**2 - .5)**2 -
      #                 r2/2 * (y - 2)**2) * (((x + 6/5)**2 + (y - 6/5)**2 - .5) *
      #                r1 * (y - 6/5) + r2 * (y - 2)) +\
      #         jnp.exp(-r1/4 * (x**2 + y**2 - 2)**2 - r2/2 * (y + 1)**2) *\
      #                 ((x**2 + y**2 - 2) * r1 * y + r2 * (y + 1))
      # truth = -jnp.concat([(dpidx/_pi)[:, None], (dpidy/_pi)[:, None]], axis=1)
      # y0 = 8/5
      # x0 = 3/2
      # _pi = jnp.exp(-r1/4 * ((x - x0)**2 + (y - 6/5)**2 - .5)**2 -
      #               r2/2 * (y - y0)**2) + 1e-20# +\
      #       # jnp.exp(-r1/4 * ((x + x0)**2 + (y - 6/5)**2 - .5)**2 -
      #       #         r2/2 * (y - y0)**2) + 1e-20
      # dpidx = jnp.exp(-r1/4 * ((x - x0)**2 + (y - 6/5)**2 - .5)**2 -
      #               r2/2 * (y - y0)**2) * ((x - x0)**2 + (y - 6/5)**2 - .5) *\
      #               r1 * (x - x0) # +\
      #         # jnp.exp(-r1/4 * ((x + x0)**2 + (y - 6/5)**2 - .5)**2 -
      #         #         r2/2 * (y - y0)**2) * ((x + x0)**2 + (y - 6/5)**2 - .5) *\
      #         #       r1 * (x + x0)
      # dpidy = jnp.exp(-r1/4 * ((x - x0)**2 + (y - 6/5)**2 - .5)**2 -
      #               r2/2 * (y - y0)**2) * (((x - x0)**2 + (y - 6/5)**2 - .5) *
      #               r1 * (y - 6/5) + r2 * (y - y0)) # +\
      #         # jnp.exp(-r1/4 * ((x + x0)**2 + (y - 6/5)**2 - .5)**2 -
      #         #         r2/2 * (y - y0)**2) * (((x + x0)**2 + (y - 6/5)**2 - .5) *
      #         #       r1 * (y - 6/5) + r2 * (y - y0))
      # truth = -jnp.concat([(dpidx/_pi)[:, None], (dpidy/_pi)[:, None]], axis=1)
      grad_x = -(x**2 + y**2 - 4) * x
      grad_y = -(x**2 + y**2 - 4) * y - 2 * (y - 1)
      truth = jnp.concat([grad_x[:, None], grad_y[:, None]], axis=1)
      truth *= a
    elif subtype == "nongradient":
      if dim != 2:
        raise Exception("nongradient case is only implemented for 2D!")
      J = jnp.array([[0, 1], [-1, 0]])
      delta = 0.5
      truth = -r3 * a + jnp.dot(r3, J) * delta
    elif subtype == "lorenz":
      if dim != 3:
        raise Exception("Lorenz dynamics is only defined for 3 dim!")
      truth = jnp.zeros((batch_size, dim))
      # _r is a parameter to change the scale of the dynamics
      _r = 4
      truth = truth.at[:, 0].set(10 * (r3[:, 1] - r3[:, 0]))
      truth = truth.at[:, 1].set(_r * r3[:, 0] * (28/_r - r3[:, 2]) - r3[:, 1])
      truth = truth.at[:, 2].set(_r * r3[:, 0] * r3[:, 1] - r3[:, 2] * 8/3)
      
    return jnp.mean((velocity - truth)**2) * dim / 2


def ot_loss_fn(
  model, dim: int, T: float, dt: float, t_batch_size: int, subtype: str,
  params: hk.Params, rng: PRNGKey, _lambda: float, batch_size: int
) -> Array:
  """Loss of the Wasserstein gradient flow, including:

  KL divergence between the normalizing flow at t=0 and the source distribution
  KL divergence between the normalizing flow at t=1 and the target distribution
  Monte-Carlo integration of the kinetic energy along the interval [0, 1]

  """
  loss = _lambda * partial(density_fit_kl_loss_fn, model, dim,
                           T)(params, rng, batch_size)
  # loss = _lambda * partial(ot_reverse_kl_loss_fn, model, dim,
  #                          T)(params, rng, batch_size)
  t_batch = jax.random.uniform(rng, (t_batch_size, ))
  # t_batch = jnp.linspace(0.05, 0.95, t_batch_size)
  for _ in range(t_batch_size):
    loss += partial(kinetic_loss_fn, model, dim, dt
                    )(params, t_batch[_], rng, batch_size // 32) / t_batch_size
    if subtype == "obstacle":
      a = 0
      loss += partial(potential_loss_fn, model, dim, a,
                      subtype)(params, t_batch[_], rng, batch_size // 32)

  return loss


def rwpo_loss_fn(
  model, dim: int, T: float, beta: float, dt: float, dx: float,
  t_batch_size: int, subtype: str, a: float, params: hk.Params, rng: PRNGKey,
  _lambda: float, batch_size: int
) -> Array:
  """Loss of the mean-field potential game
  """

  loss = _lambda * partial(reverse_kl_loss_fn, model, dim, T, beta)\
    (params, 0, rng, batch_size) +\
    partial(potential_loss_fn, model, dim, a, subtype)(params, T, rng, batch_size)
  t_batch = jax.random.uniform(rng, (t_batch_size, )) * T
  for t in t_batch:
    loss += partial(kinetic_with_score_loss_fn, model, dim, beta, dt,
                    dx)(params, t, rng, batch_size // 32) / t_batch_size * T

  return loss


def fp_loss_fn(
  model, dim: int, T: float, a: float, sigma: float, dt: float, dx: float,
  t_batch_size: int, subtype: str, params: hk.Params, rng: PRNGKey,
  _lambda: float, batch_size: int
) -> Array:
  """Loss of the mean-field potential game
  """

  beta = 4  # the initial Gaussian distribution has variance 1
  loss = _lambda * partial(reverse_kl_loss_fn, model, dim, T, beta)\
    (params, 0, rng, batch_size)
  t_batch = jax.random.uniform(rng, (t_batch_size, )) * T
  for t in t_batch:
    loss += partial(flow_matching_loss_fn, model, dim, a, sigma, subtype, dt,
                    dx)(params, t, rng, batch_size // 32) / t_batch_size * T

  return loss
