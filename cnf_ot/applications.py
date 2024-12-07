from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
from jaxtyping import Array

from cnf_ot.types import PRNGKey


def kl_loss_fn(
  model, dim: int, T: float, beta: float, params: hk.Params, cond: float,
  rng: PRNGKey, batch_size: int
) -> Array:
  """KL-divergence loss function.
  KL-divergence between the normalizing flow and the reference distribution.
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


def reverse_kl_loss_fn(
  model, dim: int, T: float, params: hk.Params, cond: float, rng: PRNGKey,
  batch_size: int
) -> Array:
  """reverse KL-divergence loss function.
  reverse KL-divergence between the normalizing flow and the reference
  distribution.
  """

  def sample_source_fn(
    seed: PRNGKey,
    sample_shape: int,
  ):

    return jax.random.normal(seed, shape=(sample_shape, dim)) *\
      jnp.ones(dim).reshape(1, -1) +\
      jnp.ones(dim).reshape(1, dim) * -3

  def sample_target_fn(
    seed: PRNGKey,
    sample_shape: int,
  ):

    return jax.random.normal(seed, shape=(sample_shape, dim)) +\
      jnp.ones(dim).reshape(1, dim) * 3

  samples1 = sample_source_fn(seed=rng, sample_shape=batch_size)
  samples2 = sample_target_fn(seed=rng, sample_shape=batch_size)
  samples = samples1 * (T - cond) / T + samples2 * cond / T
  fake_cond_ = jnp.ones((1, )) * cond
  log_prob = model.apply.log_prob(params, samples, cond=fake_cond_)
  return -log_prob.mean()


def density_fit_rkl_loss_fn(
  model, dim: int, T: float, params: hk.Params, rng: PRNGKey, batch_size: int
) -> Array:

  return partial(reverse_kl_loss_fn, model, dim, T)(
    params, 0, rng, batch_size
  ) +\
    partial(reverse_kl_loss_fn, model, dim, T)(params, T, rng, batch_size)


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


def ot_loss_fn(
  model, dim: int, T: float, dt: float, t_batch_size: int, subtype: str,
  params: hk.Params, rng: PRNGKey, _lambda: float, batch_size: int
) -> Array:
  """Loss of the Wasserstein gradient flow, including:

  KL divergence between the normalizing flow at t=0 and the source distribution
  KL divergence between the normalizing flow at t=1 and the target distribution
  Monte-Carlo integration of the kinetic energy along the interval [0, 1]

  """
  loss = _lambda * partial(density_fit_rkl_loss_fn, model, dim,
                           T)(params, rng, batch_size)
  t_batch = jax.random.uniform(rng, (t_batch_size, ))
  #t_batch = jnp.linspace(0.05, 0.95, t_batch_size)
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
  t_batch_size: int, subtype: str, params: hk.Params, rng: PRNGKey,
  _lambda: float, batch_size: int
) -> Array:
  """Loss of the mean-field potential game
  """

  a = 0
  loss = _lambda * partial(kl_loss_fn, model, dim, T, beta)(params, 0, rng, batch_size) +\
    partial(potential_loss_fn, model, dim, a, subtype)(params, T, rng, batch_size)
  t_batch = jax.random.uniform(rng, (t_batch_size, )) * T
  for t in t_batch:
    loss += partial(kinetic_with_score_loss_fn, model, dim, beta, dt,
                    dx)(params, t, rng, batch_size // 32) / t_batch_size * T

  return loss
