"""A simple example of a flow model trained to solve the Wassserstein geodesic problem."""
from functools import partial
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import optax
from absl import app, flags
from jaxtyping import Array
from tqdm import tqdm

import cnf_ot.utils as utils
from cnf_ot.models.flows import RQSFlow
from cnf_ot.types import OptState, PRNGKey

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
flags.DEFINE_integer("eval_frequency", 1000, "How often to evaluate the model.")
flags.DEFINE_integer("seed", 42, "random seed.")

flags.DEFINE_enum(
  "case", "density fit", ["density fit", "wasserstein", "mfg"], "problem type"
)

flags.DEFINE_boolean("use_64", True, "whether to use float64")
flags.DEFINE_boolean("plot", False, "whether to plot resulting model density")

flags.DEFINE_integer("dim", 1, "dimension of the base space")

FLAGS = flags.FLAGS


def gaussian_2d(
  r: jnp.ndarray,
  mean: jnp.ndarray = jnp.array([2, 3]),
  var: jnp.ndarray = jnp.array([[2, .5], [.5, 1]])
) -> jnp.ndarray:

  return jnp.exp(
    -0.5 * jnp.dot(jnp.dot((r - mean), jnp.linalg.inv(var)), (r - mean).T)
  ) / jnp.sqrt(jnp.linalg.det(2 * jnp.pi * var))


def prob_fn1_(r: jnp.ndarray, ) -> jnp.ndarray:
  """t=0"""

  R = 5.0
  var = 1.0
  rho1 = partial(gaussian_2d, mean=jnp.array([0.0, R]), var=jnp.eye(2) * var)
  rho2 = partial(gaussian_2d, mean=jnp.array([R, 0.0]), var=jnp.eye(2) * var)
  rho3 = partial(gaussian_2d, mean=jnp.array([0.0, -R]), var=jnp.eye(2) * var)
  rho4 = partial(gaussian_2d, mean=jnp.array([-R, 0.0]), var=jnp.eye(2) * var)
  return (rho1(r) + rho2(r) + rho3(r) + rho4(r)) / 4


# def sample_fn1(
#   seed: PRNGKey,
#   sample_shape,
# ):
#   """t=0"""

#   R = 5.0
#   component_indices = jax.random.choice(
#     seed, a=4, shape=(sample_shape, ), p=jnp.ones(4) / 4
#   )
#   sample_ = jnp.zeros((4, sample_shape, FLAGS.dim))
#   sample_ = sample_.at[0].set(
#     jax.random.normal(seed, shape=(sample_shape, FLAGS.dim)) + jnp.array([0.0, R])
#   )
#   sample_ = sample_.at[1].set(
#     jax.random.normal(seed, shape=(sample_shape, FLAGS.dim)) + jnp.array([R, 0.0])
#   )
#   sample_ = sample_.at[2].set(
#     jax.random.normal(seed, shape=(sample_shape, FLAGS.dim)) + jnp.array([0.0, -R])
#   )
#   sample_ = sample_.at[3].set(
#     jax.random.normal(seed, shape=(sample_shape, FLAGS.dim)) + jnp.array([-R, 0.0])
#   )

#   sample = sample_[component_indices[jnp.arange(sample_shape)],
#                    jnp.arange(sample_shape)]
#   return sample


def sample_fn1(
  seed: PRNGKey,
  sample_shape,
):
  """t=1"""

  return jax.random.normal(seed, shape=(sample_shape, FLAGS.dim)) * 3


def prob_fn2_(r: jnp.ndarray, ) -> jnp.ndarray:
  """t=1"""

  rho1 = partial(gaussian_2d, mean=jnp.array([0.0, 0.0]), var=jnp.eye(2))
  return rho1(r)


def sample_fn2(
  seed: PRNGKey,
  sample_shape,
):
  """t=1"""

  return jax.random.normal(seed, shape=(sample_shape, FLAGS.dim))


def prob_fn2__(r: jnp.ndarray, ) -> jnp.ndarray:
  """t=1"""

  rho1 = partial(gaussian_2d, mean=jnp.array([-3.0, -3.0]), var=jnp.eye(2))
  return rho1(r)


def sample_fn2_(
  seed: PRNGKey,
  sample_shape,
):
  """t=1"""

  return jax.random.normal(seed, shape=(sample_shape, FLAGS.dim)) + jnp.array(
    [-3.0, -3.0]
  ).reshape((1, FLAGS.dim))


def prob_fn3_(r: jnp.ndarray, ) -> jnp.ndarray:
  """t=0.5"""

  R = 2.5
  var = 1.0
  rho1 = partial(gaussian_2d, mean=jnp.array([0.0, R]), var=jnp.eye(2) * var)
  rho2 = partial(gaussian_2d, mean=jnp.array([R, 0.0]), var=jnp.eye(2) * var)
  rho3 = partial(gaussian_2d, mean=jnp.array([0.0, -R]), var=jnp.eye(2) * var)
  rho4 = partial(gaussian_2d, mean=jnp.array([-R, 0.0]), var=jnp.eye(2) * var)
  return (rho1(r) + rho2(r) + rho3(r) + rho4(r)) / 4


def sample_fn3(
  seed: PRNGKey,
  sample_shape,
):
  """t=0.5"""

  R = 2.5
  component_indices = jax.random.choice(
    seed, a=4, shape=(sample_shape, ), p=jnp.ones(4) / 4
  )
  sample_ = jnp.zeros((4, sample_shape, FLAGS.dim))
  sample_ = sample_.at[0].set(
    jax.random.normal(seed, shape=(sample_shape, FLAGS.dim)) +
    jnp.array([0.0, R])
  )
  sample_ = sample_.at[1].set(
    jax.random.normal(seed, shape=(sample_shape, FLAGS.dim)) +
    jnp.array([R, 0.0])
  )
  sample_ = sample_.at[2].set(
    jax.random.normal(seed, shape=(sample_shape, FLAGS.dim)) +
    jnp.array([0.0, -R])
  )
  sample_ = sample_.at[3].set(
    jax.random.normal(seed, shape=(sample_shape, FLAGS.dim)) +
    jnp.array([-R, 0.0])
  )

  sample = sample_[component_indices[jnp.arange(sample_shape)],
                   jnp.arange(sample_shape)]
  return sample


def prob_fn4_(r: jnp.ndarray, ) -> jnp.ndarray:
  """t=0.25"""

  R = 3.75
  var = 1.0
  rho1 = partial(gaussian_2d, mean=jnp.array([0.0, R]), var=jnp.eye(2) * var)
  rho2 = partial(gaussian_2d, mean=jnp.array([R, 0.0]), var=jnp.eye(2) * var)
  rho3 = partial(gaussian_2d, mean=jnp.array([0.0, -R]), var=jnp.eye(2) * var)
  rho4 = partial(gaussian_2d, mean=jnp.array([-R, 0.0]), var=jnp.eye(2) * var)
  return (rho1(r) + rho2(r) + rho3(r) + rho4(r)) / 4


def sample_fn4(
  seed: PRNGKey,
  sample_shape,
):
  """t=0.25"""

  R = 3.75
  component_indices = jax.random.choice(
    seed, a=4, shape=(sample_shape, ), p=jnp.ones(4) / 4
  )
  sample_ = jnp.zeros((4, sample_shape, FLAGS.dim))
  sample_ = sample_.at[0].set(
    jax.random.normal(seed, shape=(sample_shape, FLAGS.dim)) +
    jnp.array([0.0, R])
  )
  sample_ = sample_.at[1].set(
    jax.random.normal(seed, shape=(sample_shape, FLAGS.dim)) +
    jnp.array([R, 0.0])
  )
  sample_ = sample_.at[2].set(
    jax.random.normal(seed, shape=(sample_shape, FLAGS.dim)) +
    jnp.array([0.0, -R])
  )
  sample_ = sample_.at[3].set(
    jax.random.normal(seed, shape=(sample_shape, FLAGS.dim)) +
    jnp.array([-R, 0.0])
  )

  sample = sample_[component_indices[jnp.arange(sample_shape)],
                   jnp.arange(sample_shape)]
  return sample


def prob_fn5_(r: jnp.ndarray, ) -> jnp.ndarray:
  """t=0.75"""

  R = 1.25
  var = 1.0
  rho1 = partial(gaussian_2d, mean=jnp.array([0.0, R]), var=jnp.eye(2) * var)
  rho2 = partial(gaussian_2d, mean=jnp.array([R, 0.0]), var=jnp.eye(2) * var)
  rho3 = partial(gaussian_2d, mean=jnp.array([0.0, -R]), var=jnp.eye(2) * var)
  rho4 = partial(gaussian_2d, mean=jnp.array([-R, 0.0]), var=jnp.eye(2) * var)
  return (rho1(r) + rho2(r) + rho3(r) + rho4(r)) / 4


def sample_fn5(
  seed: PRNGKey,
  sample_shape,
):
  """t=0.75"""

  R = 1.25
  component_indices = jax.random.choice(
    seed, a=4, shape=(sample_shape, ), p=jnp.ones(4) / 4
  )
  sample_ = jnp.zeros((4, sample_shape, FLAGS.dim))
  sample_ = sample_.at[0].set(
    jax.random.normal(seed, shape=(sample_shape, FLAGS.dim)) +
    jnp.array([0.0, R])
  )
  sample_ = sample_.at[1].set(
    jax.random.normal(seed, shape=(sample_shape, FLAGS.dim)) +
    jnp.array([R, 0.0])
  )
  sample_ = sample_.at[2].set(
    jax.random.normal(seed, shape=(sample_shape, FLAGS.dim)) +
    jnp.array([0.0, -R])
  )
  sample_ = sample_.at[3].set(
    jax.random.normal(seed, shape=(sample_shape, FLAGS.dim)) +
    jnp.array([-R, 0.0])
  )

  sample = sample_[component_indices[jnp.arange(sample_shape)],
                   jnp.arange(sample_shape)]
  return sample


def potential_fn(r: jnp.ndarray, ) -> jnp.ndarray:
  return jnp.sum(r**2, axis=1) / 2


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
  print(params.keys())

  opt_state = optimizer.init(params)
  bins = 25

  # boundary condition on density
  if FLAGS.dim == 2:
    # prob_fn1 = jax.vmap(prob_fn1_)
    prob_fn1 = jax.vmap(prob_fn1_)
    prob_fn2 = jax.vmap(prob_fn2_)
    prob_fn3 = jax.vmap(prob_fn3_)
    prob_fn4 = jax.vmap(prob_fn4_)
    prob_fn5 = jax.vmap(prob_fn5_)

  # definition of loss functions
  @partial(jax.jit, static_argnames=["batch_size"])
  def kl_loss_fn(
    params: hk.Params, rng: PRNGKey, cond, batch_size: int
  ) -> Array:
    """KL-divergence between the normalizing flow and the reference distribution.
    """

    fake_cond_ = np.ones((batch_size, 1)) * cond
    samples, log_prob = model.apply.sample_and_log_prob(
      params,
      cond=fake_cond_,
      seed=rng,
      sample_shape=(batch_size, ),
    )
    return (
      log_prob - jnp.log(
        prob_fn1(samples) * (1 - cond) * (0.5 - cond) * (0.75 - cond) *
        (0.25 - cond) * 32 / 3 + prob_fn2(samples) * cond * (cond - 0.5) *
        (cond - 0.75) * (cond - 0.25) * 32 / 3 + prob_fn3(samples) * cond *
        (1 - cond) * (0.75 - cond) *
        (cond - 0.25) * 64 + prob_fn4(samples) * cond * (1 - cond) *
        (0.75 - cond) * (0.5 - cond) * 128 / 3 + prob_fn5(samples) * cond *
        (1 - cond) * (cond - 0.25) * (cond - 0.5) * 128 / 3
      )
    ).mean()

  @partial(jax.jit, static_argnames=["batch_size"])
  def reverse_kl_loss_fn(
    params: hk.Params, rng: PRNGKey, cond, batch_size: int
  ) -> Array:
    """reverse KL-divergence between the normalizing flow and the reference distribution.
    
    """

    samples1 = sample_fn1(seed=rng, sample_shape=batch_size)
    samples2 = sample_fn2(seed=rng, sample_shape=batch_size)
    # samples3 = sample_fn3(seed=rng, sample_shape=batch_size)
    # samples4 = sample_fn4(seed=rng, sample_shape=batch_size)
    # samples5 = sample_fn5(seed=rng, sample_shape=batch_size)
    samples = samples1*(1-cond)*(0.5-cond)*(0.75-cond)*(0.25-cond)*32/3 \
      + samples2*cond*(cond-0.5)*(cond-0.75)*(cond-0.25)*32/3 \
    # samples = samples1*(1-cond)*(0.5-cond)*(0.75-cond)*(0.25-cond)*32/3 \

    #   + samples2*cond*(cond-0.5)*(cond-0.75)*(cond-0.25)*32/3 \
    #   + samples3*cond*(1-cond)*(0.75-cond)*(cond-0.25)*64 \
    #   + samples4*cond*(1-cond)*(0.75-cond)*(0.5-cond)*128/3 \
    #   + samples5*cond*(1-cond)*(cond-0.25)*(cond-0.5)*128/3
    fake_cond_ = np.ones((1, )) * cond
    log_prob = model.apply.log_prob(params, samples, cond=fake_cond_)
    return -log_prob.mean()

  @partial(jax.jit, static_argnames=["batch_size"])
  def mse_loss_fn(
    params: hk.Params, rng: PRNGKey, cond, batch_size: int
  ) -> Array:
    """KL-divergence between the normalizing flow and the reference distribution.
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
        jnp.exp(log_prob) - (
          prob_fn1(samples) * (1 - cond) * (0.5 - cond) * (0.75 - cond) *
          (0.25 - cond) * 32 / 3 + prob_fn2(samples) * cond * (cond - 0.5) *
          (cond - 0.75) * (cond - 0.25) * 32 / 3 + prob_fn3(samples) * cond *
          (1 - cond) * (0.75 - cond) *
          (cond - 0.25) * 64 + prob_fn4(samples) * cond * (1 - cond) *
          (0.75 - cond) * (0.5 - cond) * 128 / 3 + prob_fn5(samples) * cond *
          (1 - cond) * (cond - 0.25) * (cond - 0.5) * 128 / 3
        )
      )**2
    ).mean()

  # # density fitting using the KL divergence, the exact form of the distribution is available
  # @partial(jax.jit, static_argnames=["batch_size"])
  # def density_fit_loss_fn(params: hk.Params, rng: PRNGKey, lambda_: float, batch_size: int) -> Array:

  #   return kl_loss_fn(params, rng, 0, batch_size) + kl_loss_fn(params, rng, 1, batch_size)

  # density fitting using the reverse KL divergence, the samples from the target distribution is available
  @partial(jax.jit, static_argnames=["batch_size"])
  def density_fit_loss_fn(
    params: hk.Params, rng: PRNGKey, lambda_: float, batch_size: int
  ) -> Array:

    return reverse_kl_loss_fn(params, rng, 0, batch_size) \
      + reverse_kl_loss_fn(params, rng, 1, batch_size) # \
    # + reverse_kl_loss_fn(params, rng, .5, batch_size) \
    # + reverse_kl_loss_fn(params, rng, .25, batch_size) \
    # + reverse_kl_loss_fn(params, rng, .75, batch_size)

  @jax.jit
  def update(params: hk.Params, rng: PRNGKey, lambda_,
             opt_state: OptState) -> Tuple[Array, hk.Params, OptState]:
    """Single SGD update step."""
    loss, grads = jax.value_and_grad(density_fit_loss_fn
                                     )(params, rng, lambda_, FLAGS.batch_size)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, new_opt_state

  # plot the distribution at t=0, 1 before training
  # plt.subplot(121)
  # if FLAGS.dim == 1:
  #   fake_cond = np.zeros((FLAGS.batch_size, 1))
  #   samples = sample_fn(params, seed=key, sample_shape=(FLAGS.batch_size, ), cond=fake_cond)
  #   plt.hist(samples[...,0], bins=bins*4, density=True)
  #   fake_cond = np.ones((FLAGS.batch_size, 1))
  #   samples = sample_fn(params, seed=key, sample_shape=(FLAGS.batch_size, ), cond=fake_cond)
  #   plt.hist(samples[...,0], bins=bins*4, density=True)
  # elif FLAGS.dim == 2:
  #   fake_cond = np.zeros((FLAGS.batch_size, 1))
  #   samples = sample_fn(params, seed=key, sample_shape=(FLAGS.batch_size, ), cond=fake_cond)
  #   plt.scatter(samples[...,0], samples[...,1], s=3, c="r")
  #   fake_cond = np.ones((FLAGS.batch_size, 1))
  #   samples = sample_fn(params, seed=key, sample_shape=(FLAGS.batch_size, ), cond=fake_cond)
  #   plt.scatter(samples[...,0], samples[...,1], s=1, c="b")

  # training loop
  loss_hist = []
  iters = tqdm(range(FLAGS.epochs))
  lambda_ = 1e2
  print_ = "rKL"
  for step in iters:
    key, rng = jax.random.split(rng)
    loss, params, opt_state = update(params, key, lambda_, opt_state)
    #lambda_ += density_fit_loss_fn(params, rng, lambda_, FLAGS.batch_size)
    loss_hist.append(loss)

    if step % FLAGS.eval_frequency == 0:
      desc_str = f"{loss=:.4e}"

      key, rng = jax.random.split(rng)
      # density fit
      if FLAGS.case == "density fit":
        if print_ == "rKL":
          KL0 = reverse_kl_loss_fn(params, rng, 0, FLAGS.batch_size)
          KL1 = reverse_kl_loss_fn(params, rng, 1, FLAGS.batch_size)
          KL2 = reverse_kl_loss_fn(params, rng, 0.5, FLAGS.batch_size)
          KL3 = reverse_kl_loss_fn(params, rng, 0.75, FLAGS.batch_size)
          KL4 = reverse_kl_loss_fn(params, rng, 0.25, FLAGS.batch_size)
        elif print_ == "KL":
          KL0 = kl_loss_fn(params, rng, 0, FLAGS.batch_size)
          KL1 = kl_loss_fn(params, rng, 1, FLAGS.batch_size)
          KL2 = kl_loss_fn(params, rng, 0.5, FLAGS.batch_size)
          KL3 = kl_loss_fn(params, rng, 0.75, FLAGS.batch_size)
          KL4 = kl_loss_fn(params, rng, 0.25, FLAGS.batch_size)
        desc_str += f" | {KL0=:.4e} "
        desc_str += f" | {KL1=:.4e} "
        desc_str += f" | {KL2=:.4e} "
        desc_str += f" | {KL3=:.4e} "
        desc_str += f" | {KL4=:.4e} "
        # MSE0 = mse_loss_fn(params, rng, 0, FLAGS.batch_size)
        # desc_str += f" | {MSE0=:.4e} "
        # MSE1 = mse_loss_fn(params, rng, 1, FLAGS.batch_size)
        # desc_str += f" | {MSE1=:.4e} "
        # MSE2 = mse_loss_fn(params, rng, .5, FLAGS.batch_size)
        # desc_str += f" | {MSE2=:.4e} "
      # wasserstein distance
      elif FLAGS.case == "wasserstein":
        KL = density_fit_loss_fn(params, rng, lambda_, FLAGS.batch_size)
        kin = loss - KL * lambda_
        desc_str += f"{KL=:.4f} | {kin=:.1f} | {lambda_=:.1f}"
      elif FLAGS.case == "mfg":
        return

      iters.set_description_str(desc_str)

  # plot the distribution at t=0, 1 after training
  plt.figure(figsize=(6, 2))
  plt.subplot(131)
  if FLAGS.dim == 1:
    fake_cond = np.zeros((FLAGS.batch_size, 1))
    samples = sample_fn(
      params, seed=key, sample_shape=(FLAGS.batch_size, ), cond=fake_cond
    )
    plt.hist(samples[..., 0], bins=bins * 4, density=True)
    fake_cond = np.ones((FLAGS.batch_size, 1))
    samples = sample_fn(
      params, seed=key, sample_shape=(FLAGS.batch_size, ), cond=fake_cond
    )
    plt.hist(samples[..., 0], bins=bins * 4, density=True)
    print(
      "kinetic energy: ",
      utils.calc_kinetic_energy(
        sample_fn, forward_fn, inverse_fn, params, rng, FLAGS.dim
      )
    )
    plt.savefig("results/fig/density_1d.pdf")
    plt.show()

    plot_1d_map = utils.plot_1d_map(
      forward_fn=forward_fn, params=params, final_mean=3
    )
    plot_1d_map

  elif FLAGS.dim == 2 and FLAGS.case == "density fit":

    plt.clf()
    plt.figure(figsize=(10, 2))
    t_array = jnp.linspace(0, 1, 5)
    cmap = plt.cm.Reds
    norm = mcolors.Normalize(vmin=-.5, vmax=1.5)

    for i in range(5):
      plt.subplot(1, 5, i + 1)
      fake_cond = np.ones((FLAGS.batch_size, 1)) * t_array[i]
      samples = sample_fn(
        params, seed=key, sample_shape=(FLAGS.batch_size, ), cond=fake_cond
      )
      plt.scatter(
        samples[..., 0],
        samples[..., 1],
        s=1,
        label=str(t_array[i]),
        color=cmap(norm(t_array[i]))
      )
      plt.legend()

    plt.savefig("results/fig/density_fit.pdf")
    plt.clf()

    # this plot routine is only for density fitting bewteen Gaussian distribution
    # utils.plot_distribution_trajectory(
    #     sample_fn,
    #     forward_fn,
    #     params,
    #     key,
    #     FLAGS.batch_size,
    #     mu1,
    #     mu2,
    #     var1,
    #     var2,
    #     fig_name=FLAGS.case+"_dist_traj"
    # )

  elif FLAGS.dim == 2 and FLAGS.case == "wasserstein":
    # this plot the distribution at t=0,1 after training
    # as well as the error of the learned mapping at t=0, 1
    # based on grid evaluation
    utils.plot_distribution_trajectory(
      sample_fn,
      forward_fn,
      params,
      key,
      FLAGS.batch_size,
      mu1,
      mu2,
      var1,
      var2,
      fig_name=FLAGS.case + "_dist_traj"
    )

    # # plot the trajectory of the distribution and velocity field
    plot_traj_and_velocity = partial(
      utils.plot_traj_and_velocity,
      sample_fn=sample_fn,
      forward_fn=forward_fn,
      inverse_fn=inverse_fn,
      params=params,
      rng=rng
    )
    plot_traj_and_velocity(quiver_size=0.01)
    plt.clf()
    plt.plot(
      jnp.linspace(5001, FLAGS.epochs, FLAGS.epochs - 5000),
      jnp.array(loss_hist[5000:])
    )
    plt.savefig("results/fig/loss_hist.pdf")

    param_count = sum(x.size for x in jax.tree.leaves(params))
    print("Network parameters: {}".format(param_count))


if __name__ == "__main__":
  app.run(main)
