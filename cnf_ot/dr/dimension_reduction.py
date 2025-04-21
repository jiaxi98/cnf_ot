import jax
import jax.numpy as jnp
import ml_collections
import yaml
from box import Box
from jax import random

from cnf_ot.types import PRNGKey
import cnf_ot.utils as utils
from cnf_ot.dr.trainers import train

def main(config_dict: ml_collections.ConfigDict):

  def generate_low_dim_data(
    key: PRNGKey, dim: int, type_: str, batch_size: int
  ):
    if type_[0] == "S":
      # sphere type
      samples = jnp.zeros((batch_size, dim))
      samples = samples.at[:, :sub_dim + 1].set(
        random.normal(key, (batch_size, sub_dim + 1))
      )
      samples /= jnp.sqrt(jnp.sum(samples**2, axis=-1))[:, None]
    elif type_[0] == "T":
      if sub_dim != 2:
        raise ValueError("Only 2D torus is supported")
      R = 5
      r = 1
      theta = random.uniform(key, (batch_size, 2), minval=0, maxval=2 * jnp.pi)
      samples = jnp.zeros((batch_size, dim))
      samples = samples.at[:, :dim].set(
        jnp.vstack(
          [
            (R + r * jnp.cos(theta[:, 1])) * jnp.sin(theta[:, 0]),
            (R + r * jnp.cos(theta[:, 1])) * jnp.cos(theta[:, 0]),
            r * jnp.sin(theta[:, 1]),
          ]
        ).T
      )
    orthog_trans = random.normal(key, (dim, dim))
    orthog_trans, _ = jnp.linalg.qr(orthog_trans)
    return samples @ orthog_trans

  config = Box(config_dict)
  dim = config.dim
  model = config.model
  rng = jax.random.PRNGKey(config.seed)
  epochs = config.train.epochs
  batch_size = config.train.batch_size
  sub_dim = int(config.type[1])
  
  data = generate_low_dim_data(rng, dim, config.type, batch_size)
  breakpoint()

  if model == "enc_dec":
    encoder, decoder, params = train(
      rng, data, dim, sub_dim, model, epochs, config
    )
    encoder_forward_fn = jax.jit(encoder.apply.forward)
  elif model == "dec_only":
    decoder, params = train(
      rng, data, dim, sub_dim, model, epochs, config
    )
  decoder_forward_fn = jax.jit(decoder.apply.forward)
  decoder_inverse_fn = jax.jit(decoder.apply.inverse)

  # this color is only for visualizing ordered samples, e.g. unit circle
  if dim <= 3:
    if config.type == "S1":
      color = random.uniform(rng, (batch_size, ))
      data = data.at[:, 0].set(jnp.sin(2 * jnp.pi * color))
      data = data.at[:, 1].set(jnp.cos(2 * jnp.pi * color))
    if model == "enc_dec":
      utils.plot_dim_reduction_reconst(
        encoder_forward_fn,
        decoder_forward_fn,
        params["encoder"],
        params["decoder"],
        dim,
        sub_dim,
        data,
      )
    elif model == "dec_only":
      utils.plot_dim_reduction_reconst(
        decoder_inverse_fn,
        decoder_forward_fn,
        params,
        params,
        dim,
        sub_dim,
        data,
      )
  # samples_ = encoder_forward_fn(params["encoder"], samples)
  breakpoint()


if __name__ == "__main__":

  with open("config/dr.yaml", "r") as file:
    config_dict = yaml.safe_load(file)
  main(config_dict)
