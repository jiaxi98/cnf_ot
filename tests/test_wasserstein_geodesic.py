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
  "num_bins", 3, "Number of bins to use in the rational-quadratic spline."
)  #
flags.DEFINE_integer("batch_size", 1024, "Batch size for training.")
flags.DEFINE_integer("test_batch_size", 20000, "Batch size for evaluation.")
flags.DEFINE_float("lr", 2e-4, "Learning rate for the optimizer.")
flags.DEFINE_integer("epochs", 20000, "Number of training steps to run.")
flags.DEFINE_integer("eval_frequency", 1000, "How often to evaluate the model.")
flags.DEFINE_integer("seed", 42, "random seed.")

flags.DEFINE_enum(
  "target", "unimodal", ["unimodal", "multimodal", "correlated"],
  "which target distribution to use"
)

flags.DEFINE_boolean('use_64', True, 'whether to use float64')
flags.DEFINE_boolean('plot', False, 'whether to plot resulting model density')

flags.DEFINE_integer("dim", 1, "dimension of the base space")

FLAGS = flags.FLAGS

def kl_ess(log_model_prob, target_prob):
  """metrics used in the tori paper."""
  weights = target_prob / jnp.exp(log_model_prob)
  Z = jnp.mean(weights)  # normalizing constant
  KL = jnp.mean(log_model_prob - jnp.log(target_prob)) + jnp.log(Z)
  ESS = jnp.sum(weights)**2 / jnp.sum(weights**2)
  return Z, KL, ESS

def gaussian_distribution_sampler(
    prng_key: int=42,
    mean: jnp.ndarray=0,
    var: jnp.ndarray=1,
    batch_size: int=256
    ) -> jnp.ndarray:
    """
    TODO: should test the sampler when debugging
    """
    
    dim = mean.shape[0]
    C = jnp.linalg.cholesky(var)
    sample = jax.random.normal(jax.random.PRNGKey(prng_key), (batch_size, dim)) @ C.T + jnp.reshape(mean, (1, dim))
    return sample

def gaussian_pdf(
    r: jnp.ndarray,
    mean: jnp.ndarray=jnp.zeros(1),
    var: jnp.ndarray=jnp.eye(1)
) -> jnp.ndarray:

    return jnp.exp(-0.5 * jnp.dot(jnp.dot((r - mean), jnp.linalg.inv(var)), (r - mean).T)) / jnp.sqrt(jnp.linalg.det(2 * jnp.pi * var))

def my_pdf_1(
    r: jnp.ndarray
    ):
  return 2 * r

def unimodal_target_unnorm_prob(theta, phi=(4.18, 5.96, 1.94), beta=1.0, dim=1):
  neg_e = sum([jnp.cos(theta[i] - phi[i]) for i in range(dim)])
  prob = jnp.exp(beta * neg_e)
  return prob

def main(_):
  jax.config.update("jax_enable_x64", FLAGS.use_64)
  np.random.seed(FLAGS.seed)
  rng = jax.random.PRNGKey(FLAGS.seed)
  optimizer = optax.adam(FLAGS.lr)

  # if FLAGS.target == "unimodal":
  #   target_unnorm_prob = unimodal_target_unnorm_prob
  # elif FLAGS.target == "multimodal":
  #   target_unnorm_prob = multimodal_target_unnorm_prob
  # elif FLAGS.target == "correlated":
  #   target_unnorm_prob = correlated_target_unnorm_prob
  # else:
  #   raise NotImplementedError

  # target_unnorm_prob = partial(
  #   target_unnorm_prob, beta=FLAGS.beta, dim=FLAGS.dim
  # )
  # target_unnorm_prob = jax.vmap(target_unnorm_prob)

  model = RQSFlow(
    event_shape=(FLAGS.dim, ),
    num_layers=FLAGS.flow_num_layers,
    hidden_sizes=[FLAGS.hidden_size] * FLAGS.mlp_num_layers,
    num_bins=FLAGS.num_bins,
    periodized=False,
  )
  model = hk.without_apply_rng(hk.multi_transform(model))
  target_unnorm_prob = jax.vmap(my_pdf_1)

  # sample_fn = jax.jit(model.apply.sample, static_argnames=['sample_shape'])

  @partial(jax.jit, static_argnames=['batch_size'])
  def kl_loss_fn(params: hk.Params, rng: PRNGKey, batch_size: int) -> Array:
    """KL-divergence between the normalizing flow and the target distribution.
    
    TODO: here, we assume the p.d.f. of the target distribution is known. We can also do the case where we only access to samples from target
    distribution and use MLE to form the loss functions.
    """
    samples, log_prob = model.apply.sample_and_log_prob(
      params, seed=rng, sample_shape=(batch_size, )
    )

    return (log_prob - jnp.log(target_unnorm_prob(samples))).mean()
  
  def w_loss_fn(params: hk.Params, rng: PRNGKey, batch_size: int) -> Array:
    """Loss of the Wasserstein gradient flow, including:

    KL divergence between the normalizing flow at t=0 and the source distribution
    KL divergence between the normalizing flow at t=1 and the target distribution
    integration of the kinetic energy along the interval [0, 1]
    """
    t = 0
    samples, log_prob = model.apply.sample_and_log_prob(
      params, t, seed=rng, sample_shape=(batch_size, )
    )

    kl_loss_source =  (log_prob - jnp.log(target_unnorm_prob(samples))).mean()
    t = 1
    samples, log_prob = model.apply.sample_and_log_prob(
      params, t, seed=rng, sample_shape=(batch_size, )
    )
    kl_loss_target =  (log_prob - jnp.log(target_unnorm_prob(samples))).mean()

    t_batch_size = 10
    t_batch = jax.random.uniform(rng, (t_batch_size, ))
    kinetic = 0.
    for _ in range(t_batch_size):
      t = t_batch[_]
      velocity = jax.grad(lambda t: model.apply.sample(params, t, seed=rng, sample_shape=(batch_size, )))(t)
      print(velocity.shape)
      kinetic += 0.5 * jnp.sum(velocity ** 2, axis=1)

    return kl_loss_source + kl_loss_target + kinetic/t_batch_size

  @partial(jax.jit, static_argnames=['batch_size'])
  def eval_fn(params: hk.Params, rng: PRNGKey, batch_size: int) -> Array:
    samples, log_prob = model.apply.sample_and_log_prob(
      params, seed=rng, sample_shape=(batch_size, )
    )
    return kl_ess(log_prob, target_unnorm_prob(samples))

  @jax.jit
  def update(params: hk.Params, rng: PRNGKey,
             opt_state: OptState) -> Tuple[Array, hk.Params, OptState]:
    """Single SGD update step."""
    loss, grads = jax.value_and_grad(kl_loss_fn)(params, rng, FLAGS.batch_size)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, new_opt_state

  key, rng = jax.random.split(rng)
  params = model.init(key, np.zeros((1, FLAGS.dim)))
  print(params.keys())

  opt_state = optimizer.init(params)

  samples = model.apply.sample(
      params, seed=rng, sample_shape=(FLAGS.batch_size, )
    )
  plt.subplot(121)
  bins = 5
  plt.hist(samples[...,0], bins=bins, density=True)
  #breakpoint()
  
  # TEST JACOBIAN
  # forward_fn = jax.jit(model.apply.forward)
  # inverse_fn = jax.jit(model.apply.inverse)
  # sample_fn = jax.jit(model.apply.sample, static_argnames=['sample_shape'])
  # r = sample_fn(params, seed=key, sample_shape=(2, ))
  # print("sampled points in physical space", r)
  # print("inverse jacobian", model.apply.inverse_jac(params, r))
  # xi = inverse_fn(params, r)
  # jac_fwd = model.apply.forward_jac(params, xi)
  # print("inverse jacobian from forward", jnp.linalg.inv(jac_fwd))

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

    #   # xi = inverse_fn(params, r)
    #   # r_inv = forward_fn(params, xi)
    #   # breakpoint()

    #   jac_inv = model.apply.inverse_jac(params, r)
    #   print("inverse jacobian", jac_inv)
    #   jac_fwd = model.apply.forward_jac(params, xi)
    #   jac_inv_from_fwd = jnp.linalg.inv(jac_fwd)
    #   print("inverse jacobian from forward", jac_inv_from_fwd)
    #   print("error", jac_inv - jac_inv_from_fwd)

    #   if FLAGS.plot:
    #     plot_torus_dist(
    #       partial(model.apply.log_prob, params), target_unnorm_prob, FLAGS.dim
    #     )

  # if FLAGS.plot:
  #   plot_torus_dist(
  #     partial(model.apply.log_prob, params), target_unnorm_prob, FLAGS.dim
  #   )
  plt.subplot(122)
  samples = model.apply.sample(
      params, seed=rng, sample_shape=(FLAGS.batch_size, )
    )
  plt.hist(samples[...,0], bins=bins*4, density=True)
  #plt.savefig("results/fig/test.pdf")
  plt.show()

if __name__ == "__main__":
  app.run(main)