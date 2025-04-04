from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import ml_collections
import optax
import yaml
from box import Box
from jax import random
from jaxtyping import Array
from matplotlib import pyplot as plt
from tqdm import tqdm

import cnf_ot.utils as utils
from cnf_ot.models.flows import RQSFlow
from cnf_ot.types import OptState, PRNGKey


def main(config_dict: ml_collections.ConfigDict):

  config = Box(config_dict)
  dim = config.general.dim
  if config.general.type == "s1":
    sub_dim = 1
  elif config.general.type == "s2":
    sub_dim = 2
  elif config.general.type == "t2":
    sub_dim = 2
  model = config.general.model
  rng = jax.random.PRNGKey(config.general.seed)
  epochs = config.train.epochs
  batch_size = config.train.batch_size

  decoder = RQSFlow(
    event_shape=(dim, ),
    num_layers=config.cnf.flow_num_layers,
    hidden_sizes=[config.cnf.hidden_size] * config.cnf.mlp_num_layers,
    num_bins=config.cnf.num_bins,
    periodized=False,
    cond_shape=(0, ),
  )
  decoder = hk.without_apply_rng(hk.multi_transform(decoder))
  decoder_forward_fn = jax.jit(decoder.apply.forward)
  if model == "enc_dec":
    # using both an encoder and a decoder
    encoder = RQSFlow(
      event_shape=(dim, ),
      num_layers=config.cnf.flow_num_layers,
      hidden_sizes=[config.cnf.hidden_size] * config.cnf.mlp_num_layers,
      num_bins=config.cnf.num_bins,
      periodized=False,
      cond_shape=(0, ),
    )
    encoder = hk.without_apply_rng(hk.multi_transform(encoder))
    encoder_forward_fn = jax.jit(encoder.apply.forward)
    encoder_rng, decoder_rng, rng = jax.random.split(rng, 3)
    # unconditional NF
    params = {"encoder": encoder.init(encoder_rng, jnp.zeros((1, dim))),
              "decoder": decoder.init(decoder_rng, jnp.zeros((1, dim)))}
  elif model == "dec_only":
    # decoder only architecture
    decoder_rng, rng = jax.random.split(rng)
    decoder_inverse_fn = jax.jit(decoder.apply.inverse)
    params = decoder.init(decoder_rng, jnp.zeros((1, dim)))
  schedule = optax.piecewise_constant_schedule(
    init_value=config.train.lr,
    boundaries_and_scales={int(b): 0.1
      for b in jnp.arange(10000, epochs, 10000)}
  )
  optimizer = optax.adam(schedule)
  opt_state = optimizer.init(params)

  def generate_low_dim_data(
    key: PRNGKey, dim: int, type_: str, batch_size: int
  ):
    if type_ == "s1":
      samples = jnp.zeros((batch_size, dim))
      samples = samples.at[:, :2].set(random.normal(key, (batch_size, 2)))
      samples /= jnp.sqrt(jnp.sum(samples**2, axis=-1))[:, None]
      return samples
    elif type_ == "s2":
      pass
    elif type_ == "t2":
      pass

  if model == "enc_dec":
    def loss_fn(params: hk.Params, x: jnp.ndarray) -> Array:
      y = encoder_forward_fn(params["encoder"], x)
      y = y.at[:, sub_dim:].set(0)
      x_reconstructed = decoder_forward_fn(params["decoder"], y)
      return jnp.mean(jnp.sum((x - x_reconstructed)**2, axis=-1))
      # NOTE: the following loss will force the map to shrink to small value
      # return jnp.mean(jnp.sum(y[:, sub_dim:]**2, axis=-1))

      # NOTE: the second constraint is too hard, forcing all the points to
      # shrink to (1, 0)
      # return jnp.mean(jnp.sum(y[:, sub_dim:]**2, axis=-1)) +\
      #   jnp.mean((jnp.sum(y**2, axis=-1) - 1)**2)
      # return jnp.mean(jnp.sum(y[:, sub_dim:]**2, axis=-1)) -\
      #   jnp.mean(jnp.log(jnp.sum(y**2, axis=-1))) * .1
  elif model == "dec_only":
    def loss_fn(params: hk.Params, x: jnp.ndarray) -> Array:
      y = decoder_inverse_fn(params, x)
      y = y.at[:, sub_dim:].set(0)
      x_reconstructed = decoder_forward_fn(params, y)
      return jnp.mean(jnp.sum((x - x_reconstructed)**2, axis=-1))

  @jax.jit
  def update(params: hk.Params,
             opt_state: OptState) -> Tuple[Array, hk.Params, OptState]:
    """Single SGD update step."""
    loss, grads = jax.value_and_grad(loss_fn)(params, samples)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, new_opt_state

  # training loop
  loss_hist = []
  iters = tqdm(range(epochs))
  samples = generate_low_dim_data(rng, dim, "s1", batch_size)
  for _ in iters:
    loss, params, opt_state = update(params, opt_state)
    loss_hist.append(loss)
    lr = schedule(_)
    desc_str = f"{lr:.2e}|{loss=:.4e}"
    iters.set_description_str(desc_str)

  plt.plot(loss_hist)
  plt.yscale("log")
  plt.savefig(f"results/fig/loss_{model}.pdf")
  plt.clf()

  # this color is only for visualizing ordered samples, e.g. unit circle
  if config.general.type == "s1":
    color = random.uniform(rng, (batch_size,))
    samples = samples.at[:, 0].set(jnp.sin(2 * jnp.pi * color))
    samples = samples.at[:, 1].set(jnp.cos(2 * jnp.pi * color))
  if model == "enc_dec":
    utils.plot_dim_reduction_reconst(
      encoder_forward_fn,
      decoder_forward_fn,
      params["encoder"],
      params["decoder"],
      sub_dim,
      samples,
      color
    )
  elif model == "dec_only":
    utils.plot_dim_reduction_reconst(
      decoder_inverse_fn,
      decoder_forward_fn,
      params,
      params,
      sub_dim,
      samples,
      color
    )
  # samples_ = encoder_forward_fn(params["encoder"], samples)
  breakpoint()


if __name__ == "__main__":

  with open("config/dr.yaml", "r") as file:
    config_dict = yaml.safe_load(file)
  main(config_dict)
