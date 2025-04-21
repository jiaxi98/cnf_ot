import jax
import jax.numpy as jnp
import ml_collections
import yaml
from absl import logging
from box import Box
from jax import random

from cnf_ot.types import PRNGKey
import cnf_ot.utils as utils
from cnf_ot.dr.trainers import train

def main(config_dict: ml_collections.ConfigDict):

  def generate_low_dim_data(
    key: PRNGKey, dim: int, type_: str, batch_size: int, rotate: bool = True
  ):
    if type_[0] == "S":
      # sphere type
      samples = jnp.zeros((batch_size, dim))
      samples = samples.at[:, :sub_dim + 1].set(
        random.normal(key, (batch_size, sub_dim + 1))
      )
      samples /= jnp.sqrt(jnp.sum(samples**2, axis=-1))[:, None]
      start = jnp.array([0.0, 0.0, 1.0])
      end = jnp.array([0.0, 0.0, -1.0])
      r = 1.5
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
      start = jnp.array([6.0, 0.0, 0.0])
      end = jnp.array([-6.0, 0.0, 0.0])
      r = 8
    if rotate:
      orthog_trans = random.normal(key, (dim, dim))
      orthog_trans, _ = jnp.linalg.qr(orthog_trans)
      samples = samples @ orthog_trans
      start = start @ orthog_trans
      end = end @ orthog_trans
    return samples, start, end, r
  
  def dynamics_path_finder(init_r: float = 5, relax: float = 1.5):
    charts = []
    encoders = []
    decoders = []
    params = []
    pos = start
    index = 0

    while True:
      print(f"Finding {index}th chart...")
      r = init_r
      while True:
        r /= relax
        chart = data[jnp.linalg.norm(data - pos, axis=-1) < r]
        encoder, decoder, params_, loss = train(
          rng, chart, dim, sub_dim, model, epochs, config
        )
        if loss[-1] < 1e-2:
          break
      charts.append(chart)
      encoders.append(encoder)
      decoders.append(decoder)
      params.append(params_)
      print(f"Chart {index} found radius {r:.2f} with loss {loss[-1]}")
      if jnp.linalg.norm(pos - end) < r:
        print(f"Chart {index} is close to end point")
        break
      pos = chart[jnp.argmin(jnp.linalg.norm(chart - end, axis=-1))]
      index += 1
      print(f"L2 dist between current pos and end: {jnp.linalg.norm(pos-end):.3f}")
      breakpoint()
    return charts, encoders, decoders

  config = Box(config_dict)
  dim = config.dim
  model = config.model
  rng = jax.random.PRNGKey(config.seed)
  epochs = config.train.epochs
  batch_size = config.train.batch_size
  sub_dim = int(config.type[1])
  
  data, start, end, r = generate_low_dim_data(rng, dim, config.type, batch_size)
  data1 = data[jnp.linalg.norm(data - start[None], axis=-1) < r]
  data2 = data[jnp.linalg.norm(data - end[None], axis=-1) < r]
  overlap = data2[jnp.linalg.norm(data2 - start[None], axis=-1) < r]
  print(f"data: {data.shape[0]}; data1: {data1.shape[0]};\
    data2: {data2.shape[0]}; overlap: {overlap.shape[0]}")
  
  dynamics_path_finder()

  if model == "enc_dec":
    encoder1, decoder1, params1, _ = train(
      rng, data1, dim, sub_dim, model, epochs, config
    )
    encoder2, decoder2, params2, _ = train(
      rng, data2, dim, sub_dim, model, epochs, config
    )
    encoders = [encoder1, encoder2]
    decoders = [decoder1, decoder2]
    params = [params1, params2]
    encoder_forward_fn = jax.jit(encoder1.apply.forward)
  elif model == "dec_only":
    decoder1, params1, _ = train(
      rng, data1, dim, sub_dim, model, epochs, config
    )
    decoder2, params2, _ = train(
      rng, data2, dim, sub_dim, model, epochs, config
    )
  decoder_forward_fn = jax.jit(decoder1.apply.forward)
  decoder_inverse_fn = jax.jit(decoder1.apply.inverse)

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
        params1["encoder"],
        params1["decoder"],
        dim,
        sub_dim,
        data,
      )
    elif model == "dec_only":
      utils.plot_dim_reduction_reconst(
        decoder_inverse_fn,
        decoder_forward_fn,
        params1,
        params1,
        dim,
        sub_dim,
        data,
      )
  # samples_ = encoder_forward_fn(params["encoder"], samples)
  utils.find_mfd_path(
    encoders, decoders, params, data1, data2, overlap, sub_dim,
    start, end, "S2_path.png"
  )
  breakpoint()


if __name__ == "__main__":

  with open("config/dr.yaml", "r") as file:
    config_dict = yaml.safe_load(file)
  main(config_dict)
