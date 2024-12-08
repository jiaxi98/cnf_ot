"""A simple example of a flow model trained to solve the Wassserstein geodesic problem."""
from functools import partial
from typing import Iterator, Optional, Tuple

import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import optax
import tensorflow_datasets as tfds
from absl import app, flags, logging
from jaxtyping import Array
from tqdm import tqdm

import cnf_ot.utils as utils
from cnf_ot.flows import RQSFlow
from cnf_ot.types import Batch, OptState, PRNGKey

flags.DEFINE_integer(
  "flow_num_layers", 2, "Number of layers to use in the flow."
)
flags.DEFINE_integer(
  "mlp_num_layers", 2, "Number of layers to use in the MLP conditioner."
)  # 2
flags.DEFINE_integer(
  "hidden_size", 16, "Hidden size of the MLP conditioner."
)  # 64
flags.DEFINE_integer(
  "num_bins", 5, "Number of bins to use in the rational-quadratic spline."
)  # 20
flags.DEFINE_integer("batch_size", 2048, "Batch size for training.")
flags.DEFINE_integer("test_batch_size", 20000, "Batch size for evaluation.")
flags.DEFINE_float("lr", 1e-3, "Learning rate for the optimizer.")
flags.DEFINE_integer("epochs", 30000, "Number of training steps to run.")
flags.DEFINE_integer("eval_frequency", 100, "How often to evaluate the model.")
flags.DEFINE_integer("seed", 42, "random seed.")

flags.DEFINE_enum(
  "case", "wasserstein", ["density fit", "wasserstein", "mfg"], "problem type"
)

flags.DEFINE_boolean("use_64", True, "whether to use float64")
flags.DEFINE_boolean("plot", False, "whether to plot resulting model density")

flags.DEFINE_integer("dim", 2, "dimension of the base space")

FLAGS = flags.FLAGS
T = 1


def gaussian_2d(
  r: jnp.ndarray,
  mean: jnp.ndarray = jnp.array([2, 3]),
  var: jnp.ndarray = jnp.array([[2, .5], [.5, 1]])
) -> jnp.ndarray:

  return jnp.exp(
    -0.5 * jnp.dot(jnp.dot((r - mean), jnp.linalg.inv(var)), (r - mean).T)
  ) / jnp.sqrt(jnp.linalg.det(2 * jnp.pi * var))


def gaussian_mixture_2d(r: jnp.ndarray, ) -> jnp.ndarray:

  R = 10
  rho1 = partial(gaussian_2d, mean=jnp.array([0.0, R]), var=jnp.eye(2))
  rho2 = partial(gaussian_2d, mean=jnp.array([R, 0.0]), var=jnp.eye(2))
  rho3 = partial(gaussian_2d, mean=jnp.array([0.0, -R]), var=jnp.eye(2))
  rho4 = partial(gaussian_2d, mean=jnp.array([-R, 0.0]), var=jnp.eye(2))
  return (rho1(r) + rho2(r) + rho3(r) + rho4(r)) / 4


def sample_source_fn(
  dim: int,
  seed: PRNGKey,
  sample_shape,
):

  return jax.random.normal(seed, shape=(sample_shape, dim)) *\
    jnp.ones(dim).reshape(1, -1) +\
    jnp.ones(dim).reshape(1, dim) * -3

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
  dim: int,
  seed: PRNGKey,
  sample_shape,
):

  return jax.random.normal(seed, shape=(sample_shape, dim)) +\
    jnp.ones(dim).reshape(1, dim) * 3


def main(_):

  # model initialization
  jax.config.update("jax_enable_x64", FLAGS.use_64)
  np.random.seed(FLAGS.seed)
  rng = jax.random.PRNGKey(FLAGS.seed)
  optimizer = optax.adam(FLAGS.lr)

  model = RQSFlow(
    event_shape=(FLAGS.dim, ),
    num_layers=FLAGS.flow_num_layers,
    hidden_sizes=[FLAGS.hidden_size] * FLAGS.mlp_num_layers,
    num_bins=FLAGS.num_bins,
    periodized=False,
  )
  model = hk.without_apply_rng(hk.multi_transform(model))
  forward_fn = jax.jit(model.apply.forward)
  inverse_fn = jax.jit(model.apply.inverse)
  sample_fn = jax.jit(model.apply.sample, static_argnames=["sample_shape"])
  log_prob_fn = jax.jit(model.apply.log_prob)
  key, rng = jax.random.split(rng)
  params = model.init(key, np.zeros((1, FLAGS.dim)), np.zeros((1, )))
  opt_state = optimizer.init(params)

  # boundary condition on density
  # NOTE: need to simultaneous modified the sample_source_fn as well
  source_prob = jax.vmap(
    partial(
      jax.scipy.stats.multivariate_normal.pdf,
      mean=jnp.ones(FLAGS.dim) * -3,
      cov=jnp.eye(FLAGS.dim)
    )
  )
  target_prob = jax.vmap(
    partial(
      jax.scipy.stats.multivariate_normal.pdf,
      mean=jnp.ones(FLAGS.dim) * 3,
      cov=jnp.eye(FLAGS.dim)
    )
  )

  # definition of loss functions
  @partial(jax.jit, static_argnames=["batch_size"])
  def kl_loss_fn(
    params: hk.Params, rng: PRNGKey, cond, batch_size: int
  ) -> Array:
    """KL-divergence loss function.
    KL-divergence between the normalizing flow and the reference distribution.
    """

    fake_cond_ = np.ones((batch_size, 1)) * cond
    samples, log_prob = model.apply.sample_and_log_prob(
      params,
      cond=fake_cond_,
      seed=rng,
      sample_shape=(batch_size, ),
    )
    return (
      log_prob -
      jnp.log(source_prob(samples) * (1 - cond) + target_prob(samples) * cond)
    ).mean()

  @partial(jax.jit, static_argnames=["batch_size"])
  def reverse_kl_loss_fn(
    params: hk.Params, rng: PRNGKey, cond, batch_size: int
  ) -> Array:
    """reverse KL-divergence loss function.
    reverse KL-divergence between the normalizing flow and the reference
    distribution.
    """

    samples1 = sample_source_fn(
      dim=FLAGS.dim, seed=rng, sample_shape=batch_size
    )
    samples2 = sample_target_fn(
      dim=FLAGS.dim, seed=rng, sample_shape=batch_size
    )
    samples = samples1 * (1 - cond) + samples2 * cond
    fake_cond_ = np.ones((1, )) * cond
    log_prob = model.apply.log_prob(params, samples, cond=fake_cond_)
    return -log_prob.mean()

  @partial(jax.jit, static_argnames=["batch_size"])
  def mse_loss_fn(
    params: hk.Params, rng: PRNGKey, cond, batch_size: int
  ) -> Array:
    """MSE between the normalizing flow and the reference distribution.
    """

    fake_cond_ = jnp.ones((batch_size, 1)) * cond
    samples, log_prob = model.apply.sample_and_log_prob(
      params,
      cond=fake_cond_,
      seed=rng,
      sample_shape=(batch_size, ),
    )
    return (
      (
        jnp.exp(log_prob) -
        (source_prob(samples) * (1 - cond) + target_prob(samples) * cond)
      )**2
    ).mean()

  # # kinetic energy based on auto-differentiation
  # @partial(jax.jit, static_argnames=["batch_size"])
  # def kinetic_loss_fn(t: float, params: hk.Params, rng: PRNGKey, batch_size: int) -> Array:
  #     """Kinetic energy along the trajectory at time t
  #     """
  #     fake_cond_ = np.ones((batch_size, 1)) * t
  #     samples = sample_fn(params, seed=rng, sample_shape=(batch_size, ), cond=fake_cond_)
  #     xi = inverse_fn(params, samples, fake_cond_)
  #     velocity = jax.jacfwd(partial(forward_fn, params, xi))(fake_cond_)
  #     # velocity.shape = [batch_size, 2, batch_size, 1]
  #     # velocity[jnp.arange(batch_size),:,jnp.arange(batch_size),0].shape = [batch_size, 2]
  #     return jnp.mean(velocity[jnp.arange(batch_size),:,jnp.arange(batch_size),0]**2) * FLAGS.dim / 2

  # kinetic energy based on finite difference
  @partial(jax.jit, static_argnames=["batch_size"])
  def kinetic_loss_fn(
    t: float, params: hk.Params, rng: PRNGKey, batch_size: int
  ) -> Array:
    """Kinetic energy along the trajectory at time t
      """
    dt = 0.01
    fake_cond_ = np.ones((batch_size, 1)) * (t - dt / 2)
    # NOTE: this implementation may have some caveat as two samples may not be
    # exactly corresponding
    r1 = sample_fn(
      params, seed=rng, sample_shape=(batch_size, ), cond=fake_cond_
    )
    fake_cond_ = np.ones((batch_size, 1)) * (t + dt / 2)
    r2 = sample_fn(
      params, seed=rng, sample_shape=(batch_size, ), cond=fake_cond_
    )
    velocity = (r2 - r1) / dt  # velocity.shape = [batch_size, 2]

    return jnp.mean(velocity**2) * FLAGS.dim / 2

  # density fitting using the reverse KL divergence, the samples from the target distribution is available
  @partial(jax.jit, static_argnames=["batch_size"])
  def density_fit_kl_loss_fn(
    params: hk.Params, rng: PRNGKey, lambda_: float, batch_size: int
  ) -> Array:

    return kl_loss_fn(params, rng, 0,
                      batch_size) + kl_loss_fn(params, rng, 1, batch_size)

  @partial(jax.jit, static_argnames=["batch_size"])
  def density_fit_rkl_loss_fn(
    params: hk.Params, rng: PRNGKey, lambda_: float, batch_size: int
  ) -> Array:

    return reverse_kl_loss_fn(params, rng, 0, batch_size) + reverse_kl_loss_fn(
      params, rng, 1, batch_size
    )

  @partial(jax.jit, static_argnames=["batch_size"])
  def wasserstein_loss_fn(
    params: hk.Params, rng: PRNGKey, lambda_: float, batch_size: int
  ) -> Array:
    """Loss of the Wasserstein gradient flow, including:

    KL divergence between the normalizing flow at t=0 and the source distribution
    KL divergence between the normalizing flow at t=1 and the target distribution
    Monte-Carlo integration of the kinetic energy along the interval [0, 1]

    NOTE: adding loss function corresponds to acc will significantly slower the computation

    TODO: one caveat of the coding here is that we do not further split the 
    rng for sampling from CNF in kl_loss and kinetic_loss. 
    """

    def potential_fn(r: jnp.ndarray, ) -> jnp.ndarray:
      return 50 * jnp.exp(-jnp.sum(r**2, axis=1) / 2)

    def potential_loss_fn(
      params: hk.Params, rng: PRNGKey, cond, batch_size: int
    ) -> Array:

      fake_cond_ = np.ones((batch_size, 1)) * cond
      samples, _ = model.apply.sample_and_log_prob(
        params,
        cond=fake_cond_,
        seed=rng,
        sample_shape=(batch_size, ),
      )
      return potential_fn(samples).mean()

    # def acc_loss_fn(
    #   t: float, params: hk.Params, rng: PRNGKey, batch_size: int
    # ) -> Array:
    #   """acceleration energy along the trajectory at time t, used for regularization
    #   originally introduced to enforce the linear map for linear OT problem
    #   """
    #   fake_cond_ = np.ones((batch_size, 1)) * t
    #   samples = sample_fn(
    #     params, seed=rng, sample_shape=(batch_size, ), cond=fake_cond_
    #   )
    #   xi = inverse_fn(params, samples, fake_cond_)
    #   velocity = jax.jacfwd(partial(forward_fn, params, xi))(fake_cond_)
    #   # velocity.shape = [batch_size, DIM, batch_size, 1]
    #   # velocity[jnp.arange(batch_size),:,jnp.arange(batch_size),0].shape = [batch_size, 2]

    #   dt = 0.01
    #   fake_cond_ = np.ones((batch_size, 1)) * (t + dt)
    #   samples = sample_fn(
    #     params, seed=rng, sample_shape=(batch_size, ), cond=fake_cond_
    #   )
    #   xi = inverse_fn(params, samples, fake_cond_)
    #   velocity_ = jax.jacfwd(partial(forward_fn, params, xi))(fake_cond_)
    #   #weight = .01
    #   return jnp.mean(
    #     velocity[jnp.arange(batch_size), :,
    #              jnp.arange(batch_size), 0] -
    #     velocity_[jnp.arange(batch_size), :,
    #               jnp.arange(batch_size), 0]
    #   ) * FLAGS.dim / 2

    loss = lambda_ * density_fit_rkl_loss_fn(params, rng, lambda_, batch_size)
    # loss = lambda_ * (density_fit_rkl_loss_fn(params, rng, lambda_, batch_size) -
    #   density_fit_kl_loss_fn(params, rng, lambda_, batch_size))
    t_batch_size = 10
    t_batch = jax.random.uniform(rng, (t_batch_size, ))
    #t_batch = jnp.linspace(0.05, 0.95, t_batch_size)
    for _ in range(t_batch_size):
      loss += kinetic_loss_fn(
        t_batch[_], params, rng, batch_size // 32
      ) / t_batch_size
      # loss += potential_loss_fn(
      #   params, rng, t_batch[_], batch_size // 32
      # )
      # + acc_loss_fn(t_batch[_], params, rng, batch_size//32)/t_batch_size

    return loss

  @jax.jit
  def update(params: hk.Params, rng: PRNGKey, lambda_,
             opt_state: OptState) -> Tuple[Array, hk.Params, OptState]:
    """Single SGD update step."""
    loss, grads = jax.value_and_grad(wasserstein_loss_fn
                                     )(params, rng, lambda_, FLAGS.batch_size)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, new_opt_state

  @jax.jit
  def pretrain_update(
    params: hk.Params, rng: PRNGKey, lambda_, opt_state: OptState
  ) -> Tuple[Array, hk.Params, OptState]:
    """Single SGD update step."""
    loss, grads = jax.value_and_grad(density_fit_rkl_loss_fn
                                     )(params, rng, lambda_, FLAGS.batch_size)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, new_opt_state

  # pretraining loop
  if False:
    loss_hist = []
    lambda_ = 1
    iters = tqdm(range(FLAGS.epochs))
    for step in iters:
      key, rng = jax.random.split(rng)
      loss, params, opt_state = pretrain_update(params, key, lambda_, opt_state)
      loss_hist.append(loss)

      if step % FLAGS.eval_frequency == 0:
        desc_str = f"{loss=:.4e}"

        key, rng = jax.random.split(rng)
        if FLAGS.case == "density fit":
          KL0 = kl_loss_fn(params, rng, 0, FLAGS.batch_size)
          desc_str += f" | {KL0=:.4e} "
          KL1 = kl_loss_fn(params, rng, 1, FLAGS.batch_size)
          desc_str += f" | {KL1=:.4e} "

        iters.set_description_str(desc_str)

    plt.clf()
    plt.plot(loss_hist[5000:])
    plt.savefig("results/fig/pretrain_loss_hist.pdf")

  # NOTE: here we test that: with the same rng, CNF-generated samples
  # corresponds to the same prior point in the latent space. Therefore
  # our implementation of finite difference approximation of the velocity
  # is unbiased.
  if False:
    batch_size = 64
    fake_cond_ = np.ones((batch_size, 1)) * 0.8
    r1 = sample_fn(
      params, seed=rng, sample_shape=(batch_size, ), cond=fake_cond_
    )
    xi1 = inverse_fn(params, r1, jnp.ones(1) * .8)
    fake_cond_ = np.ones((batch_size, 1)) * 0.38
    r2 = sample_fn(
      params, seed=rng, sample_shape=(batch_size, ), cond=fake_cond_
    )
    xi2 = inverse_fn(params, r1, jnp.ones(1) * .38)

  # training loop
  loss_hist = []
  iters = tqdm(range(FLAGS.epochs))
  lambda_ = 1000
  print(f"Solving optimal transport in {FLAGS.dim}D...")
  for step in iters:
    key, rng = jax.random.split(rng)
    loss, params, opt_state = update(params, key, lambda_, opt_state)
    #lambda_ += density_fit_loss_fn(params, rng, lambda_, FLAGS.batch_size)
    loss_hist.append(loss)

    if step % FLAGS.eval_frequency == 0:
      desc_str = f"{loss=:.4e}"

      key, rng = jax.random.split(rng)
      KL = density_fit_kl_loss_fn(params, rng, lambda_, FLAGS.batch_size)
      desc_str += f"{KL=:.4f} | {lambda_=:.1f}"
      iters.set_description_str(desc_str)

  plt.plot(
    jnp.linspace(5001, FLAGS.epochs, FLAGS.epochs - 5000),
    jnp.array(loss_hist[5000:])
  )
  plt.savefig("results/fig/loss_hist.pdf")

  param_count = sum(x.size for x in jax.tree.leaves(params))
  print("Network parameters: {}".format(param_count))

  print(
    "kinetic energy with more samples: ",
    utils.calc_kinetic_energy(
      sample_fn, params, rng, batch_size=65536, t_size=10000, dim=FLAGS.dim
    )
  )

  print(
    "kinetic energy with less samples: ",
    utils.calc_kinetic_energy(
      sample_fn, params, rng, batch_size=4096, t_size=1000, dim=FLAGS.dim
    )
  )
  breakpoint()

  if FLAGS.dim == 2:
    # utils.plot_distribution_trajectory(
    #   sample_fn,
    #   forward_fn,
    #   params,
    #   key,
    #   FLAGS.batch_size,
    #   mu1,
    #   mu2,
    #   var1,
    #   var2,
    #   fig_name=FLAGS.case + "_dist_traj"
    # )

    # # plot the trajectory of the distribution and velocity field
    # plot_traj_and_velocity = partial(
    #   utils.plot_traj_and_velocity,
    #   sample_fn=sample_fn,
    #   forward_fn=forward_fn,
    #   inverse_fn=inverse_fn,
    #   params=params,
    #   rng=rng,
    #   t_array=t_array,
    # )
    # plot_traj_and_velocity(quiver_size=0.01)

    # this is for Gaussian mixture to Gaussian transport
    # R = 5
    # r_ = jnp.vstack([
    #   jnp.array([0.0, R]), jnp.array([0.0, -R]),
    #   jnp.array([R, 0.0]), jnp.array([-R, 0.0]),
    #   jnp.array([0.6*R, 0.8*R]), jnp.array([0.6*R, -0.8*R]),
    #   jnp.array([-0.6*R, 0.8*R]), jnp.array([-0.6*R, -0.8*R])])

    r_ = jnp.vstack(
      [
        jnp.array([-2.0, -2.0]),
        jnp.array([-2.0, -3.0]),
        jnp.array([-2.0, -4.0]),
        jnp.array([-3.0, -2.0]),
        jnp.array([-3.0, -3.0]),
        jnp.array([-3.0, -4.0]),
        jnp.array([-4.0, -2.0]),
        jnp.array([-4.0, -3.0]),
        jnp.array([-4.0, -4.0])
      ]
    )
    t_array = jnp.linspace(0, T, 20)
    utils.plot_density_and_trajectory(
      forward_fn,
      inverse_fn,
      log_prob_fn,
      params=params,
      r_=r_,
      t_array=t_array,
    )


if __name__ == "__main__":
  app.run(main)
