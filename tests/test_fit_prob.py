"""A simple example of a flow model trained to solve the Wassserstein geodesic problem."""
from functools import partial
from typing import Iterator, Optional, Tuple

import haiku as hk
import jax
import distrax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tensorflow_datasets as tfds
from absl import app, flags, logging
from jaxtyping import Array
from tqdm import tqdm

from src.flows import RQSFlow
from src.types import Batch, OptState, PRNGKey

flags.DEFINE_integer(
  "flow_num_layers", 1, "Number of layers to use in the flow."
)
flags.DEFINE_integer(
  "mlp_num_layers", 2, "Number of layers to use in the MLP conditioner."
)
flags.DEFINE_integer("hidden_size", 64, "Hidden size of the MLP conditioner.")
flags.DEFINE_integer(
  "num_bins", 10, "Number of bins to use in the rational-quadratic spline."
)  #
flags.DEFINE_integer("batch_size", 8192, "Batch size for training.")
flags.DEFINE_integer("test_batch_size", 20000, "Batch size for evaluation.")
flags.DEFINE_float("lr", 2e-4, "Learning rate for the optimizer.")
flags.DEFINE_integer("epochs", 50000, "Number of training steps to run.")
flags.DEFINE_integer("eval_frequency", 1000, "How often to evaluate the model.")
flags.DEFINE_integer("seed", 42, "random seed.")

flags.DEFINE_enum(
  "target", "unimodal", ["unimodal", "multimodal", "correlated"],
  "which target distribution to use"
)

flags.DEFINE_boolean('use_64', True, 'whether to use float64')
flags.DEFINE_boolean('plot', False, 'whether to plot resulting model density')

flags.DEFINE_integer("dim", 2, "dimension of the base space")

FLAGS = flags.FLAGS


def kl_ess(log_model_prob, target_prob):
  """metrics used in the tori paper."""
  weights = target_prob / jnp.exp(log_model_prob)
  Z = jnp.mean(weights)  # normalizing constant
  KL = jnp.mean(log_model_prob - jnp.log(target_prob)) + jnp.log(Z)
  ESS = jnp.sum(weights)**2 / jnp.sum(weights**2)
  return Z, KL, ESS


def gaussian_2d(r: jnp.ndarray) -> jnp.ndarray:

  #mean = jnp.reshape(jnp.array([2, 3]), (1,2))
  mean = jnp.array([2, 3])
  var = jnp.array([[2, .5], [.5, 1]])
  return jnp.exp(
    -0.5 * jnp.dot(jnp.dot((r - mean), jnp.linalg.inv(var)), (r - mean).T)
  ) / jnp.sqrt(jnp.linalg.det(2 * jnp.pi * var))


def main(_):
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

  #target_pdf = distrax.Uniform(low=5, high=8).prob            # 1D uniform, not a good target as the support is not the whole axis
  #target_pdf = distrax.Normal(loc=3, scale=2).prob             # 1D Gaussian
  #target_pdf = distrax.Normal(loc=jnp.array([1, 2]), scale=jnp.array([[2,.5]])).prob        # 2D Gaussian
  target_prob = jax.vmap(gaussian_2d)

  sample_fn = jax.jit(model.apply.sample, static_argnames=['sample_shape'])

  @partial(jax.jit, static_argnames=['batch_size'])
  def kl_loss_fn(params: hk.Params, rng: PRNGKey, batch_size: int) -> Array:
    """KL-divergence between the normalizing flow and the target distribution.
    
    TODO: here, we assume the p.d.f. of the target distribution is known. We can also do the case where we only access to samples from target
    distribution and use MLE to form the loss functions.
    """
    fake_cond_ = np.zeros((batch_size, 1))
    samples, log_prob = model.apply.sample_and_log_prob(
      params,
      cond=fake_cond_,
      seed=rng,
      sample_shape=(batch_size, ),
    )

    return (log_prob - jnp.log(target_prob(samples))).mean()
    # 2D distrax.Normal samples
    #return (log_prob - jnp.log(jnp.prod(target_prob(samples), axis=(1,2)))).mean()

  @partial(jax.jit, static_argnames=['batch_size'])
  def eval_fn(params: hk.Params, rng: PRNGKey, batch_size: int) -> Array:
    fake_cond_ = np.zeros((batch_size, 1))
    samples, log_prob = model.apply.sample_and_log_prob(
      params, cond=fake_cond_, seed=rng, sample_shape=(batch_size, )
    )

    return kl_ess(log_prob, target_prob(samples))
    # 2D distrax.Normal samples
    #return kl_ess(log_prob, jnp.prod(target_prob(samples), axis=(1,2)))

  @jax.jit
  def update(params: hk.Params, rng: PRNGKey,
             opt_state: OptState) -> Tuple[Array, hk.Params, OptState]:
    """Single SGD update step."""
    loss, grads = jax.value_and_grad(kl_loss_fn)(params, rng, FLAGS.batch_size)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, new_opt_state

  key, rng = jax.random.split(rng)
  params = model.init(key, np.zeros((1, FLAGS.dim)), np.zeros((1, 1)))
  print(params.keys())

  opt_state = optimizer.init(params)

  fake_cond = np.zeros((FLAGS.batch_size, 1))
  samples = sample_fn(
    params, seed=key, sample_shape=(FLAGS.batch_size, ), cond=fake_cond
  )
  #   print("inverse jacobian", model.apply.inverse_jac(params, r, fake_cond))
  #   xi = inverse_fn(params, r, fake_cond)
  #   jac_fwd = model.apply.forward_jac(params, xi, fake_cond)
  #   print("inverse jacobian from forward", jnp.linalg.inv(jac_fwd))

  #   def log_prob_fn(params, r):
  #     fake_cond = jnp.zeros((r.shape[0], 1))
  #     return model.apply.log_prob(params, r, fake_cond)
  plt.subplot(121)
  if FLAGS.dim == 1:
    bins = 5
    plt.hist(samples[..., 0], bins=bins * 4, density=True)
  elif FLAGS.dim == 2:
    plt.scatter(samples[..., 0], samples[..., 1], s=1)

  loss_hist = []
  iters = tqdm(range(FLAGS.epochs))
  for step in iters:
    key, rng = jax.random.split(rng)
    loss, params, opt_state = update(params, key, opt_state)
    #desc_str = f"{loss=:.2E}"
    #iters.set_description(desc_str)
    loss_hist.append(loss)

    if step % FLAGS.eval_frequency == 0:
      desc_str = f"{loss=:.2f}"

      key, rng = jax.random.split(rng)
      Z, KL, ESS = eval_fn(params, key, FLAGS.test_batch_size)
      ESS = ESS / FLAGS.test_batch_size * 100
      desc_str += f" | {Z=:.2f} | {KL=:.2E} | {ESS=:.2f}%"
      iters.set_description_str(desc_str)

  plt.subplot(122)
  samples = sample_fn(
    params, seed=key, sample_shape=(FLAGS.batch_size, ), cond=fake_cond
  )
  if FLAGS.dim == 1:
    bins = 5
    plt.hist(samples[..., 0], bins=bins * 4, density=True)
    print('min of samples', jnp.min(samples[..., 0]))
    print('max of samples', jnp.max(samples[..., 0]))
    x = jnp.linspace(jnp.min(samples[..., 0]), jnp.max(samples[..., 0]), 1000)
    pdf = target_pdf(x)
    plt.plot(x, pdf, label='ground_truth')
    plt.legend()
  elif FLAGS.dim == 2:
    plt.scatter(samples[..., 0], samples[..., 1], s=1)
  plt.savefig("results/fig/Gaussian2D.pdf")
  plt.show()


if __name__ == "__main__":
  app.run(main)
