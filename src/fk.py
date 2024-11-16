"""A simple example of a flow model trained to solve the Fokker-Planck
equation."""
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
  "case", "mfg", ["density fit", "wasserstein", "mfg"], "problem type"
)

flags.DEFINE_boolean("use_64", True, "whether to use float64")
flags.DEFINE_boolean("plot", False, "whether to plot resulting model density")

flags.DEFINE_integer("dim", 2, "dimension of the base space")

FLAGS = flags.FLAGS

T = 1
a = 1 # drift coeff
sigma = 1/2 # diffusion coeff sigma = D^2/2

def gaussian_2d(
  r: jnp.ndarray,
  mean: jnp.ndarray = jnp.array([2, 3]),
  var: jnp.ndarray = jnp.array([[2, .5], [.5, 1]])
) -> jnp.ndarray:

  return jnp.exp(
    -0.5 * jnp.dot(jnp.dot((r - mean), jnp.linalg.inv(var)), (r - mean).T)
  ) / jnp.sqrt(jnp.linalg.det(2 * jnp.pi * var))


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
  if FLAGS.dim == 1:
    if FLAGS.case == "density fit":
      # Gaussian source
      # source_prob = jax.vmap(distrax.Normal(loc=0, scale=1).prob)

      # Gaussian mixture source
      def source_prob_(r):
        return distrax.Normal(loc=0, scale=1).prob(r) * .3 + distrax.Normal(
          loc=4, scale=1
        ).prob(r) * .7

      source_prob = jax.vmap(source_prob_)
      target_prob = jax.vmap(distrax.Normal(loc=2, scale=1).prob)

    else:
      source_prob = jax.vmap(distrax.Normal(loc=0, scale=2).prob)
      # target_prob is useless
      target_prob = jax.vmap(distrax.Normal(loc=2, scale=1).prob)

  elif FLAGS.dim == 2:
    var1 = 1
    var2 = 1
    mu1 = jnp.array([-5, -5])
    mu2 = jnp.array([5, 5])
    # # linear transport calculation case
    # source_prob = jax.vmap(partial(gaussian_2d, mean=mu1, var=jnp.eye(2)*var1))
    # target_prob = jax.vmap(partial(gaussian_2d, mean=mu2, var=jnp.eye(2)*var2))
    # multi-to-one case
    source_prob = jax.vmap(
      partial(gaussian_2d, mean=jnp.array([0.0, 0.0]), var=4 * jnp.eye(2))
    )
    target_prob = jax.vmap(
      partial(
        gaussian_2d, 
        mean=jnp.array([0.0, 0.0]), 
        var=jnp.eye(2) * (jnp.exp(-2 * a * T) * (4 - 1/2/a) + 1 / 2 / a)
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
      jnp.log(source_prob(samples) * (T - cond)/T + target_prob(samples) * cond/T)
    ).mean()

  @partial(jax.jit, static_argnames=["batch_size"])
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
  
  # score-modified kinetic energy based on finite difference
  @partial(jax.jit, static_argnames=["batch_size"])
  def flow_matching_loss_fn(
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
    score = jnp.zeros((batch_size, FLAGS.dim))
    dx = 0.01
    for i in range(FLAGS.dim):
      dr = jnp.zeros((1, FLAGS.dim))
      dr = dr.at[0, i].set(dx/2)
      log_p1 = log_prob_fn(params, r3 + dr, cond=jnp.ones(1) * t)
      log_p2 = log_prob_fn(params, r3 - dr, cond=jnp.ones(1) * t)
      score = score.at[:, i].set((log_p1 - log_p2) / dx)
    velocity += score * sigma
    truth = -r3 * a
    return jnp.mean((velocity - truth)**2) * FLAGS.dim / 2

  @partial(jax.jit, static_argnames=["batch_size"])
  def fk_loss_fn(
    params: hk.Params, rng: PRNGKey, lambda_: float, batch_size: int
  ) -> Array:
    """Loss of the mean-field potential game
    """

    loss = lambda_ * kl_loss_fn(params, rng, 0, batch_size)
    t_batch_size = 20  # 10
    t_batch = jax.random.uniform(rng, (t_batch_size, )) * T
    for t in t_batch:
      loss += flow_matching_loss_fn(
        t, params, rng, batch_size // 64
      ) / t_batch_size

    return loss
  
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

  @jax.jit
  def update(params: hk.Params, rng: PRNGKey, lambda_,
             opt_state: OptState) -> Tuple[Array, hk.Params, OptState]:
    """Single SGD update step."""
    loss, grads = jax.value_and_grad(fk_loss_fn
                                     )(params, rng, lambda_, FLAGS.batch_size)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, new_opt_state

  # training loop
  loss_hist = []
  iters = tqdm(range(FLAGS.epochs))
  lambda_ = 1000
  for step in iters:
    key, rng = jax.random.split(rng)
    loss, params, opt_state = update(params, key, lambda_, opt_state)
    #lambda_ += density_fit_loss_fn(params, rng, lambda_, FLAGS.batch_size)
    loss_hist.append(loss)

    if step % FLAGS.eval_frequency == 0:
      terminal_loss = kl_loss_fn(params, rng, 1, FLAGS.batch_size)
      desc_str = f"{loss=:.4e}|{terminal_loss:.4e}"
      iters.set_description_str(desc_str)

  if FLAGS.dim == 2:
    # this plot the distribution at t=0,1 after training
    # as well as the error of the learned mapping at t=0, 1
    # based on grid evaluation
    plt.clf()
    plt.plot(
      jnp.linspace(5001, FLAGS.epochs, FLAGS.epochs - 5000),
      jnp.array(loss_hist[5000:])
    )
    plt.savefig("results/fig/loss_hist.pdf")

    param_count = sum(x.size for x in jax.tree.leaves(params))
    print("Network parameters: {}".format(param_count))
    print(potential_loss_fn(params, rng, 1, FLAGS.batch_size))
    breakpoint()

    r_ = jnp.vstack(
      [jnp.array([-1.0, -1.0]), jnp.array([-1.0, -0.0]),
       jnp.array([-1.0, 1.0]), jnp.array([0.0, -1.0]),
       jnp.array([0.0, 0.0]), jnp.array([0.0, 1.0]),
       jnp.array([1.0, -1.0]), jnp.array([1.0, 0.0]),
       jnp.array([1.0, 1.0])]
    )
    r_ = r_ * 3
    t_array = jnp.linspace(0, T, 20)
    utils.plot_density_and_trajectory(
      forward_fn,
      inverse_fn,
      log_prob_fn,
      params=params,
      r_ = r_,
      t_array=t_array,
    )
    breakpoint()


if __name__ == "__main__":
  app.run(main)
