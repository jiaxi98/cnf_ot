"""Reproduce the experiment in the paper:
Rezende, et al. Normalizing Flows on Tori and Spheres (2020)."""
from functools import partial
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import app, flags
from jaxtyping import Array
from tqdm import tqdm

from dpw.flows import RQSFlow
from dpw.types import OptState, PRNGKey
from dpw.visualize import plot_torus_dist

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
flags.DEFINE_integer("batch_size", 256, "Batch size for training.")
flags.DEFINE_integer("test_batch_size", 20000, "Batch size for evaluation.")
flags.DEFINE_float("lr", 2e-4, "Learning rate for the optimizer.")
flags.DEFINE_integer("epochs", 20000, "Number of training steps to run.")
flags.DEFINE_integer("eval_frequency", 2000, "How often to evaluate the model.")
flags.DEFINE_integer("seed", 42, "random seed.")

flags.DEFINE_float(
  "beta", 1.0, "Inverse temperature for the target distribution."
)
flags.DEFINE_enum(
  "target", "unimodal", ["unimodal", "multimodal", "correlated"],
  "which target distribution to use"
)

flags.DEFINE_boolean('plot', False, 'whether to plot resulting model density')

flags.DEFINE_integer("dim", 2, "dimension of the torus")

FLAGS = flags.FLAGS


def kl_ess(log_model_prob, target_prob):
  """metrics used in the tori paper."""
  weights = target_prob / jnp.exp(log_model_prob)
  Z = jnp.mean(weights)  # normalizing constant
  KL = jnp.mean(log_model_prob - jnp.log(target_prob)) + jnp.log(Z)
  ESS = jnp.sum(weights)**2 / jnp.sum(weights**2)
  return Z, KL, ESS


# UNNORMALIZED TARGET DISTRIBUTION on T2
def unimodal_target_unnorm_prob(theta, phi=(4.18, 5.96, 1.94), beta=1.0, dim=2):
  neg_e = sum([jnp.cos(theta[i] - phi[i]) for i in range(dim)])
  prob = jnp.exp(beta * neg_e)
  return prob


def multimodal_target_unnorm_prob(
  theta,
  phis=(
    (0.21, 2.85, 4.18),
    (1.89, 6.18, 2.14),
    (3.77, 1.56, 3.77),
  ),
  beta=1.0,
  dim=2,
):
  prob = sum(
    [unimodal_target_unnorm_prob(theta, phi_i, beta, dim) for phi_i in phis]
  ) / len(phis)
  return prob


def correlated_target_unnorm_prob(theta, phi=1.94, beta=1.0, dim=2):
  neg_e = jnp.cos(sum(theta) - phi)
  prob = jnp.exp(beta * neg_e)
  return prob


def main(_):
  np.random.seed(FLAGS.seed)
  rng = jax.random.PRNGKey(FLAGS.seed)
  optimizer = optax.adam(FLAGS.lr)

  if FLAGS.target == "unimodal":
    target_unnorm_prob = unimodal_target_unnorm_prob
  elif FLAGS.target == "multimodal":
    target_unnorm_prob = multimodal_target_unnorm_prob
  elif FLAGS.target == "correlated":
    target_unnorm_prob = correlated_target_unnorm_prob
  else:
    raise NotImplementedError

  target_unnorm_prob = partial(
    target_unnorm_prob, beta=FLAGS.beta, dim=FLAGS.dim
  )
  target_unnorm_prob = jax.vmap(target_unnorm_prob)

  model = RQSFlow(
    event_shape=(FLAGS.dim, ),
    num_layers=FLAGS.flow_num_layers,
    hidden_sizes=[FLAGS.hidden_size] * FLAGS.mlp_num_layers,
    num_bins=FLAGS.num_bins,
    periodized=True,
  )
  model = hk.without_apply_rng(hk.multi_transform(model))

  # sample_fn = jax.jit(model.apply.sample, static_argnames=['sample_shape'])

  @partial(jax.jit, static_argnames=['batch_size'])
  def loss_fn(params: hk.Params, rng: PRNGKey, batch_size: int) -> Array:
    """KL-divergence between the model and the target distribution."""
    samples, log_prob = model.apply.sample_and_log_prob(
      params, seed=rng, sample_shape=(batch_size, )
    )
    return (log_prob - jnp.log(target_unnorm_prob(samples))).mean()

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
    loss, grads = jax.value_and_grad(loss_fn)(params, rng, FLAGS.batch_size)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, new_opt_state

  key, rng = jax.random.split(rng)
  params = model.init(key, np.zeros((1, FLAGS.dim)))
  print(params.keys())
  breakpoint()

  opt_state = optimizer.init(params)

  # TEST JACOBIAN
  forward_fn = jax.jit(model.apply.forward)
  inverse_fn = jax.jit(model.apply.inverse)
  sample_fn = jax.jit(model.apply.sample, static_argnames=['sample_shape'])
  r = sample_fn(params, seed=key, sample_shape=(2, ))
  print("sampled points in physical space", r)
  print("inverse jacobian", model.apply.inverse_jac(params, r))
  xi = inverse_fn(params, r)
  jac_fwd = model.apply.forward_jac(params, xi)
  print("inverse jacobian from forward", jnp.linalg.inv(jac_fwd))

  iters = tqdm(range(FLAGS.epochs))
  for step in iters:
    key, rng = jax.random.split(rng)
    loss, params, opt_state = update(params, key, opt_state)

    if step % FLAGS.eval_frequency == 0:
      desc_str = f"{loss=:.2f}"

      key, rng = jax.random.split(rng)
      Z, KL, ESS = eval_fn(params, key, FLAGS.test_batch_size)
      ESS = ESS / FLAGS.test_batch_size * 100
      desc_str += f" | {Z=:.2f} | {KL=:.2E} | {ESS=:.2f}%"
      iters.set_description_str(desc_str)

      # xi = inverse_fn(params, r)
      # r_inv = forward_fn(params, xi)
      # breakpoint()

      jac_inv = model.apply.inverse_jac(params, r)
      print("inverse jacobian", jac_inv)
      jac_fwd = model.apply.forward_jac(params, xi)
      jac_inv_from_fwd = jnp.linalg.inv(jac_fwd)
      print("inverse jacobian from forward", jac_inv_from_fwd)
      print("error", jac_inv - jac_inv_from_fwd)

      if FLAGS.plot:
        plot_torus_dist(
          partial(model.apply.log_prob, params), target_unnorm_prob, FLAGS.dim
        )

  if FLAGS.plot:
    plot_torus_dist(
      partial(model.apply.log_prob, params), target_unnorm_prob, FLAGS.dim
    )


if __name__ == "__main__":
  app.run(main)
