"""A simple example of a flow model trained to solve the Wassserstein geodesic problem."""
from functools import partial
from typing import Iterator, Optional, Tuple

import haiku as hk
import jax
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
flags.DEFINE_integer("batch_size", 1024, "Batch size for training.")
flags.DEFINE_integer("test_batch_size", 20000, "Batch size for evaluation.")
flags.DEFINE_float("lr", 2e-4, "Learning rate for the optimizer.")
flags.DEFINE_integer("epochs", 2000, "Number of training steps to run.")
flags.DEFINE_integer("eval_frequency", 100, "How often to evaluate the model.")
flags.DEFINE_integer("seed", 42, "random seed.")

flags.DEFINE_enum(
  "target", "unimodal", ["unimodal", "multimodal", "correlated"],
  "which target distribution to use"
)

flags.DEFINE_boolean('use_64', True, 'whether to use float64')
flags.DEFINE_boolean('plot', False, 'whether to plot resulting model density')

flags.DEFINE_integer("dim", 2, "dimension of the base space")

FLAGS = flags.FLAGS

# def kl_ess(log_model_prob, target_prob):
#   """metrics used in the tori paper."""
#   weights = target_prob / jnp.exp(log_model_prob)
#   Z = jnp.mean(weights)  # normalizing constant
#   KL = jnp.mean(log_model_prob - jnp.log(target_prob)) + jnp.log(Z)
#   ESS = jnp.sum(weights)**2 / jnp.sum(weights**2)
#   return Z, KL, ESS

# def gaussian_distribution_sampler(
#     prng_key: int=42,
#     mean: jnp.ndarray=0,
#     var: jnp.ndarray=1,
#     batch_size: int=256
#     ) -> jnp.ndarray:
#     """
#     TODO: should test the sampler when debugging
#     """
    
#     dim = mean.shape[0]
#     C = jnp.linalg.cholesky(var)
#     sample = jax.random.normal(jax.random.PRNGKey(prng_key), (batch_size, dim)) @ C.T + jnp.reshape(mean, (1, dim))
#     return sample

def gaussian_2d(
    r: jnp.ndarray,
    mean: jnp.ndarray=jnp.array([2, 3]),
    var: jnp.ndarray=jnp.array([[2, .5], [.5, 1]])
) -> jnp.ndarray:

    return jnp.exp(-0.5 * jnp.dot(jnp.dot((r - mean), jnp.linalg.inv(var)), (r - mean).T)) / jnp.sqrt(jnp.linalg.det(2 * jnp.pi * var))

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
  source_prob = jax.vmap(partial(gaussian_2d, mean=jnp.array([-1,-1]), var=jnp.eye(2)))
  target_prob = jax.vmap(partial(gaussian_2d, mean=jnp.array([3,3]), var=jnp.eye(2)))

  forward_fn = jax.jit(model.apply.forward)
  inverse_fn = jax.jit(model.apply.inverse)
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
    loss = (log_prob - jnp.log(source_prob(samples))).mean()

    fake_cond_ = np.ones((batch_size, 1))
    samples, log_prob = model.apply.sample_and_log_prob(
      params,
      cond=fake_cond_,
      seed=rng,
      sample_shape=(batch_size, ),
    )
    loss += (log_prob - jnp.log(target_prob(samples))).mean()
    return loss

  @partial(jax.jit, static_argnames=['batch_size'])
  def kinetic_loss_fn(t: float, params: hk.Params, rng: PRNGKey, batch_size: int) -> Array:

    #batch_size = 2
    fake_cond_ = np.ones((batch_size, 1)) * t
    samples = sample_fn(params, seed=rng, sample_shape=(batch_size, ), cond=fake_cond_)
    #fake_cond_ = np.ones((2, 1)) * t
    #sample = sample_fn(params, seed=rng, sample_shape=(2, ), cond=fake_cond_)
    xi = inverse_fn(params, samples, fake_cond_)
    velocity = jax.jacfwd(partial(forward_fn, params, xi))(fake_cond_)
    #print(velocity.shape)
    #rint(velocity)
    #breakpoint()
    weight = .01
    return jnp.mean(0.5 * velocity**2) * 2 * batch_size * weight
  
  @partial(jax.jit, static_argnames=['batch_size'])
  def w_loss_fn(params: hk.Params, rng: PRNGKey, batch_size: int) -> Array:
    """Loss of the Wasserstein gradient flow, including:

    KL divergence between the normalizing flow at t=0 and the source distribution
    KL divergence between the normalizing flow at t=1 and the target distribution
    integration of the kinetic energy along the interval [0, 1]
    """
    
    loss = kl_loss_fn(params, rng, batch_size)
    t_batch_size = 10
    t_batch = jax.random.uniform(rng, (t_batch_size, ))
    for _ in range(t_batch_size):
      loss += kinetic_loss_fn(t_batch[_], params, rng, batch_size//32)

    return loss

  # @partial(jax.jit, static_argnames=['batch_size'])
  # def eval_fn(params: hk.Params, rng: PRNGKey, batch_size: int) -> Array:
  #   samples, log_prob = model.apply.sample_and_log_prob(
  #     params, seed=rng, sample_shape=(batch_size, )
  #   )
  #   return kl_ess(log_prob, target_unnorm_prob(samples))

  @jax.jit
  def update(params: hk.Params, rng: PRNGKey,
             opt_state: OptState) -> Tuple[Array, hk.Params, OptState]:
    """Single SGD update step."""
    loss, grads = jax.value_and_grad(w_loss_fn)(params, rng, FLAGS.batch_size)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, new_opt_state

  key, rng = jax.random.split(rng)
  params = model.init(key, np.zeros((1, FLAGS.dim)), np.zeros((1, 1)))
  print(params.keys())

  opt_state = optimizer.init(params)

  plt.subplot(121)
  if FLAGS.dim == 1:
    bins = 5
    plt.hist(samples[...,0], bins=bins*4, density=True)
  elif FLAGS.dim == 2:
    fake_cond = np.zeros((FLAGS.batch_size, 1))
    samples = sample_fn(params, seed=key, sample_shape=(FLAGS.batch_size, ), cond=fake_cond)
    plt.scatter(samples[...,0], samples[...,1], s=3, c='r')
    fake_cond = np.ones((FLAGS.batch_size, 1))
    samples = sample_fn(params, seed=key, sample_shape=(FLAGS.batch_size, ), cond=fake_cond)
    plt.scatter(samples[...,0], samples[...,1], s=1, c='b')

  loss_hist = []
  iters = tqdm(range(FLAGS.epochs))
  for step in iters:
    key, rng = jax.random.split(rng)
    loss, params, opt_state = update(params, key, opt_state)
    loss_hist.append(loss)

    if step % FLAGS.eval_frequency == 0:
      desc_str = f"{loss=:.2f}"

      key, rng = jax.random.split(rng)
      KL = kl_loss_fn(params, rng, FLAGS.batch_size)
      kin = kinetic_loss_fn(0.5, params, rng, FLAGS.batch_size)
      desc_str += f" | {KL=:.2f} | {kin=:.2f}"
      iters.set_description_str(desc_str)

  plt.subplot(122)
  if FLAGS.dim == 1:
    bins = 5
    plt.hist(samples[...,0], bins=bins*4, density=True)
  elif FLAGS.dim == 2:
    fake_cond = np.zeros((FLAGS.batch_size, 1))
    samples = sample_fn(params, seed=key, sample_shape=(FLAGS.batch_size, ), cond=fake_cond)
    plt.scatter(samples[...,0], samples[...,1], s=1, c='r')
    fake_cond = np.ones((FLAGS.batch_size, 1))
    samples = sample_fn(params, seed=key, sample_shape=(FLAGS.batch_size, ), cond=fake_cond)
    plt.scatter(samples[...,0], samples[...,1], s=1, c='b')
    plt.savefig('results/fig/w1.pdf')

  plt.clf()
  t_array = [00, 0.2, 0.4, 0.6, 0.8, 1.0]
  i = 1
  for t in t_array:
    plt.subplot(3, 2, i)
    fake_cond = np.ones((FLAGS.batch_size, 1)) * t
    samples = sample_fn(params, seed=key, sample_shape=(FLAGS.batch_size, ), cond=fake_cond)
    plt.scatter(samples[...,0], samples[...,1], s=1)
    i += 1
  plt.savefig('results/fig/w2.pdf')
  breakpoint()


if __name__ == "__main__":
  app.run(main)