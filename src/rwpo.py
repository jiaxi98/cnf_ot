"""A simple example of a flow model trained to solve the regularized
Wasserstein proximal operator ."""
# TODO: code cleanup
# the key and rng are used alternatively throughout the code
# the name of the repo will be changed to cnf_ot
from functools import partial
from typing import Iterator, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import distrax
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import optax
import tensorflow_datasets as tfds
from absl import app, flags, logging
from jaxtyping import Array
from tqdm import tqdm

from src.flows import RQSFlow
from src.types import Batch, OptState, PRNGKey
import src.utils as utils

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
  "case", "rwpo", ["density fit", "wasserstein", "rwpo"], "problem type"
)

flags.DEFINE_boolean("use_64", True, "whether to use float64")
flags.DEFINE_boolean("plot", False, "whether to plot resulting model density")

flags.DEFINE_integer("dim", 10, "dimension of the base space")

FLAGS = flags.FLAGS

# parameter for rwpo exp
T = 1
beta = 1
pot_mode = "quadratic" # quadratic, double-well


# test the gaussian source
def sample_g_source_fn(
  seed: PRNGKey,
  sample_shape,
):
  """
  According to the exact solution from LQR, the initial condition is given by 
  rho_0 \sim N(0, 2(T+1)I), we let T=1 here so the variance is 4.
  """
  
  return jax.random.normal(seed, shape=(sample_shape, FLAGS.dim)) * 2


def sample_target_fn(
  seed: PRNGKey,
  sample_shape,
):

  return jax.random.normal(seed, shape=(sample_shape, FLAGS.dim))


def potential_fn(r: jnp.ndarray, ) -> jnp.ndarray:
  # quadratic potential
  if pot_mode == "quadratic":
    return jnp.sum(r**2, axis=1) / 2
  # double-well potential
  elif pot_mode == "double-well":
    return (jnp.linalg.norm(r - jnp.ones(FLAGS.dim).reshape(1, -1), axis=1) *
      jnp.linalg.norm(r + jnp.ones(FLAGS.dim).reshape(1, -1), axis=1) / 2) ** 2


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
  source_prob = jax.vmap(
    partial(
      jax.scipy.stats.multivariate_normal.pdf, 
      mean=jnp.zeros(FLAGS.dim), 
      cov=jnp.eye(FLAGS.dim) * 2 / beta * (T + 1)
    )
  )
  target_prob = jax.vmap(
    partial(
      jax.scipy.stats.multivariate_normal.pdf, 
      mean=jnp.zeros(FLAGS.dim), 
      cov=jnp.eye(FLAGS.dim) * 2 / beta,
    )
  )

  # definition of loss functions
  # @partial(jax.jit, static_argnames=["batch_size"])
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
      jnp.log(source_prob(samples) * (T - cond)/T + target_prob(samples) * cond/T)
    ).mean()

  # @partial(jax.jit, static_argnames=["batch_size"])
  def reverse_kl_loss_fn(
    params: hk.Params, rng: PRNGKey, cond, batch_size: int
  ) -> Array:
    """reverse KL-divergence loss function.
    reverse KL-divergence between the normalizing flow and the reference
    distribution.
    """

    samples1 = sample_g_source_fn(seed=rng, sample_shape=batch_size)
    samples2 = sample_target_fn(seed=rng, sample_shape=batch_size)
    samples = samples1 * (T - cond)/T + samples2 * cond/T
    fake_cond_ = np.ones((1, )) * cond
    log_prob = model.apply.log_prob(params, samples, cond=fake_cond_)
    return -log_prob.mean()

  # @partial(jax.jit, static_argnames=["batch_size"])
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
  # @partial(jax.jit, static_argnames=["batch_size"])
  def kinetic_loss_fn(
    t: float, params: hk.Params, rng: PRNGKey, batch_size: int
  ) -> Array:
    """Kinetic energy along the trajectory at time t
      """
    dt = 0.01
    fake_cond_ = np.ones((batch_size, 1)) * (t - dt / 2)
    # TODO: work out why this one does not work
    # r1 = sample_fn(
    #   params, seed=rng, sample_shape=(batch_size, ), cond=fake_cond_
    # )
    # xi = inverse_fn(params, r1, jnp.ones(1) * (t - dt / 2))
    # r2 = forward_fn(params, xi, jnp.ones(1) * (t - dt / 2))

    # NOTE: this implementation may have some caveat as two samples may not be
    # exactly corresponding
    r1 = sample_fn(
      params, seed=rng, sample_shape=(batch_size, ), cond=fake_cond_
    )
    fake_cond_ = np.ones((batch_size, 1)) * (t + dt / 2)
    r2 = sample_fn(
      params, seed=rng, sample_shape=(batch_size, ), cond=fake_cond_
    )
    velocity = (r2 - r1) / dt     # velocity.shape = [batch_size, 2]

    return jnp.mean(velocity**2) * FLAGS.dim / 2
  
  # score-modified kinetic energy based on finite difference
  # @partial(jax.jit, static_argnames=["batch_size"])
  def kinetic_with_score_loss_fn(
    t: float, params: hk.Params, rng: PRNGKey, batch_size: int
  ) -> Array:
    """Kinetic energy along the trajectory at time t, notice that this contains
    not only the velocity but also the score function
    """
    dt = 0.01
    fake_cond_ = np.ones((batch_size, 1)) * (t - dt / 2)
    r1 = sample_fn(
      params, seed=rng, sample_shape=(batch_size, ), cond=fake_cond_
    )
    fake_cond_ = np.ones((batch_size, 1)) * (t + dt / 2)
    r2 = sample_fn(
      params, seed=rng, sample_shape=(batch_size, ), cond=fake_cond_
    )
    fake_cond_ = np.ones((batch_size, 1)) * t
    r3 = sample_fn(
      params, seed=rng, sample_shape=(batch_size, ), cond=fake_cond_
    )
    velocity = (r2 - r1) / dt
    # velocity += jax.jacfwd(partial(log_prob_fn, params, cond=jnp.ones(1) * t))(
    #   r3)[jnp.arange(batch_size),jnp.arange(batch_size)]/beta
    # finite difference approximation of the score function
    score = jnp.zeros((batch_size, FLAGS.dim))
    dx = 0.01
    for i in range(FLAGS.dim):
      dr = jnp.zeros((1, FLAGS.dim))
      dr = dr.at[0, i].set(dx/2)
      log_p1 = log_prob_fn(params, r3 + dr, cond=jnp.ones(1) * t)
      log_p2 = log_prob_fn(params, r3 - dr, cond=jnp.ones(1) * t)
      score = score.at[:, i].set((log_p1 - log_p2) / dx)
    velocity += score/beta
    return jnp.mean(velocity**2) * FLAGS.dim / 2

  # density fitting using the reverse KL divergence, the samples from the target distribution is available
  # @partial(jax.jit, static_argnames=["batch_size"])
  def density_fit_kl_loss_fn(
    params: hk.Params, rng: PRNGKey, lambda_: float, batch_size: int
  ) -> Array:

    return kl_loss_fn(params, rng, 0,
                      batch_size) + kl_loss_fn(params, rng, T, batch_size)

  # @partial(jax.jit, static_argnames=["batch_size"])
  def density_fit_rkl_loss_fn(
    params: hk.Params, rng: PRNGKey, lambda_: float, batch_size: int
  ) -> Array:

    return reverse_kl_loss_fn(params, rng, 0, batch_size) + reverse_kl_loss_fn(
      params, rng, T, batch_size
    )

  # @partial(jax.jit, static_argnames=["batch_size"])
  def rwpo_loss_fn(
    params: hk.Params, rng: PRNGKey, lambda_: float, batch_size: int
  ) -> Array:
    """Loss of the mean-field potential game
    """

    loss = lambda_ * kl_loss_fn(params, rng, 0, batch_size) \
      + potential_loss_fn(params, rng, T, batch_size)
    t_batch_size = 1
    t_batch = jax.random.uniform(rng, (t_batch_size, )) * T
    for t in t_batch:
      loss += kinetic_with_score_loss_fn(
        t, params, rng, batch_size // 32
      ) / t_batch_size * T

    return loss

  @jax.jit
  def update(params: hk.Params, rng: PRNGKey, lambda_,
             opt_state: OptState) -> Tuple[Array, hk.Params, OptState]:
    """Single SGD update step."""
    loss, grads = jax.value_and_grad(rwpo_loss_fn
                                     )(params, rng, lambda_, FLAGS.batch_size)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, new_opt_state

  # training loop
  loss_hist = []
  iters = tqdm(range(FLAGS.epochs))
  lambda_ = 200
  print(f"Solving regularized Wasserstein proximal in {FLAGS.dim}D...")
  for step in iters:
    key, rng = jax.random.split(rng)
    loss, params, opt_state = update(params, key, lambda_, opt_state)
    #lambda_ += density_fit_loss_fn(params, rng, lambda_, FLAGS.batch_size)
    loss_hist.append(loss)

    if step % FLAGS.eval_frequency == 0:
      desc_str = f"{loss=:.4e}"

      key, rng = jax.random.split(rng)
      # wasserstein distance
      if FLAGS.case == "wasserstein":
        KL = density_fit_rkl_loss_fn(params, rng, lambda_, FLAGS.batch_size)
        kin = loss - KL * lambda_
        desc_str += f"{KL=:.4f} | {kin=:.1f} | {lambda_=:.1f}"
      elif FLAGS.case == "rwpo":
        # KL = reverse_kl_loss_fn(params, rng, 0, FLAGS.batch_size)
        KL = kl_loss_fn(params, rng, 0, FLAGS.batch_size)
        pot = potential_loss_fn(params, rng, T, FLAGS.batch_size)
        kin = loss - KL * lambda_ - pot
        desc_str += f"{KL=:.4f} | {pot=:.2f} | {kin=:.2f} | {lambda_=:.1f}"

      iters.set_description_str(desc_str)

  plt.plot(
    jnp.linspace(5001, FLAGS.epochs, FLAGS.epochs - 5000),
    jnp.array(loss_hist[5000:])
  )
  plt.savefig("results/fig/loss_hist.pdf")
  param_count = sum(x.size for x in jax.tree.leaves(params))
  print("Network parameters: {}".format(param_count))
  e_kin = T * utils.calc_score_kinetic_energy(
    sample_fn,
    log_prob_fn,
    params,
    T,
    beta,
    FLAGS.dim,
    rng,
  )
  e_pot = potential_loss_fn(params, rng, T, 65536)
  print("kinetic energy: ", e_kin)
  print("potential energy: ", e_pot)
  # NOTE: this is the true value for quadratic potential and Gaussian IC
  true_val = FLAGS.dim * (1 + jnp.log(T + 1)) / beta
  print("total energy: {:.3f}|relative err: {:.3e}".format(
    e_kin + e_pot, (e_kin + e_pot - true_val) / true_val * 100
  ))
  breakpoint()
  if FLAGS.dim == 2:
    # this plot the distribution at t=0,1 after training
    # as well as the error of the learned mapping at t=0, 1
    # based on grid evaluation

    r_ = jnp.vstack(
      [jnp.array([-1.0, -1.0]), jnp.array([-1.0, -0.0]),
       jnp.array([-1.0, 1.0]), jnp.array([0.0, -1.0]),
       jnp.array([0.0, 0.0]), jnp.array([0.0, 1.0]),
       jnp.array([1.0, -1.0]), jnp.array([1.0, 0.0]),
       jnp.array([1.0, 1.0])]
    )
    r_ = r_ * 3
    # utils.plot_score(log_prob_fn, params, r_)
    # breakpoint()

    t_array = jnp.linspace(0, T, 20)
    utils.plot_density_and_trajectory(
      forward_fn,
      inverse_fn,
      log_prob_fn,
      params=params,
      r_ = r_,
      t_array=t_array,
    )

  # plot the 1D rwpo example：
  # plot the histogram w.r.t. the ground truth solution
  # plot the velocity at several time step v.s. the ground truth
  if FLAGS.case == "rwpo" and FLAGS.dim == 1:
    plt.clf()
    t = jnp.linspace(0, 1, 6)
    for i in range(2):
      for j in range(3):
        plt.subplot(2,3,i*3+j+1)
        fake_cond = np.ones((FLAGS.test_batch_size, 1)) * t[i*3+j]
        samples = sample_fn(params, seed=key, sample_shape=(FLAGS.test_batch_size, ), cond=fake_cond)
        plt.hist(samples[...,0], bins=bins*4, density=True)
        x = jnp.linspace(-5, 5, 1000)
        rho = jax.vmap(distrax.Normal(loc=0, scale=jnp.sqrt(beta*2*(2-t[i*3+j]))).prob)(x)
        plt.plot(x, rho, label=r"$\rho_*$")
        plt.legend()
    plt.savefig("results/fig/rwpo.pdf")
    plt.clf()

    kinetic_err = []
    t_array = jnp.linspace(0, 1, 101)
    batch_size = 1000
    for t in t_array:
      fake_cond_ = np.ones((batch_size, 1)) * t
      samples = sample_fn(params, seed=rng, sample_shape=(batch_size, ), cond=fake_cond_)
      xi = inverse_fn(params, samples, fake_cond_)
      velocity = jax.jacfwd(partial(forward_fn, params, xi))(fake_cond_)[jnp.arange(batch_size),:,jnp.arange(batch_size),0]
      ground_truth = -jnp.sqrt(1/8/(2-t)) * samples
      kinetic_err.append(jnp.mean((velocity - ground_truth)**2))
      plt.figure(figsize=(4, 2))
      plt.scatter(samples, velocity, c="b", label="compute", s=.1)
      plt.scatter(samples, ground_truth, c="r", label="ground truth", s=.1)
      plt.legend()
      plt.title("t = {:.2f}".format(t))
      plt.savefig("results/fig/{:.2f}.pdf".format(t))
      plt.clf()
    plt.plot(t_array, kinetic_err, label=r"$\left\| \dot{x} - \dot{x}_* \right\|^2$")
    plt.legend()
    plt.savefig("results/fig/rwpo_kin.pdf")
    breakpoint()

    batch_size = FLAGS.batch_size
    loss = potential_loss_fn(params, rng, 1, batch_size)
    t_batch_size = 100 # 10
    t_batch = jax.random.uniform(rng, (t_batch_size, ))
    for _ in range(t_batch_size):
        loss += kinetic_loss_fn(t_batch[_], params, rng, batch_size//32)/t_batch_size
    print("loss: {:.4f}".format(loss))


if __name__ == "__main__":
  app.run(main)
