import jax
import jax.numpy as jnp
import ml_collections
import yaml
from box import Box
from jax import random

import cnf_ot.utils as utils
from cnf_ot.dr.trainers import dynamics_path_finder
from cnf_ot.types import PRNGKey


def main(config_dict: ml_collections.ConfigDict):

  def generate_low_dim_data(
    key: PRNGKey, dim: int, type_: str, batch_size: int, rotate: bool = True
  ):
    """Generate data lying on a low-dimensional manifold.
    
    Returns:
      samples: The generated samples.
      start: The starting point of the path.
      end: The ending point of the path.
      r: The radius of the neighborhood.
    """

    if type_[0] == "S":
      # sphere type
      samples = jnp.zeros((batch_size, dim))
      samples = samples.at[:, :sub_dim + 1].set(
        random.normal(key, (batch_size, sub_dim + 1))
      )
      samples /= jnp.sqrt(jnp.sum(samples**2, axis=-1))[:, None]
      start = jnp.zeros((dim,))
      start = start.at[0].set(1)
      end = jnp.zeros((dim,))
      end = end.at[0].set(-1)
      r = 1.5
    elif type_[0] == "T":
      if sub_dim != 2:
        raise ValueError("Only 2D torus is supported")
      R = 5
      r = 1
      theta = random.uniform(key, (batch_size, 2), minval=0, maxval=2 * jnp.pi)
      samples = jnp.zeros((batch_size, dim))
      samples = samples.at[:, :3].set(
        jnp.vstack(
          [
            (R + r * jnp.cos(theta[:, 1])) * jnp.sin(theta[:, 0]),
            (R + r * jnp.cos(theta[:, 1])) * jnp.cos(theta[:, 0]),
            r * jnp.sin(theta[:, 1]),
          ]
        ).T
      )
      start = jnp.zeros((dim,))
      start = start.at[0].set(R + r)
      end = jnp.zeros((dim,))
      end = end.at[0].set(-R - r)
      r = 8
    orthog_trans = jnp.eye(dim)
    if rotate:
      orthog_trans = random.normal(key, (dim, dim))
      orthog_trans, _ = jnp.linalg.qr(orthog_trans)
    samples = samples @ orthog_trans
    start = start @ orthog_trans
    end = end @ orthog_trans
    return samples, start, end, r, orthog_trans

  config = Box(config_dict)
  dim = config.dim
  rng = jax.random.PRNGKey(config.seed)
  batch_size = config.train.batch_size
  sub_dim = int(config.type[1])

  data, start, end, _, orthog_trans = generate_low_dim_data(
    rng, dim, config.type, batch_size
  )
  charts, pos, radius, encoders, decoders, params = dynamics_path_finder(
    config_dict, data, start, end
  )
  path = utils.find_long_mfd_path(
    encoders, decoders, params, charts, pos, radius, sub_dim, start, end, data,
    f"{config.type}_path.png"
  )
  acc = utils.check_path_accuracy(path @ orthog_trans.T, config.type)
  print(f"Accuracy: {acc:.4f}")
  breakpoint()


if __name__ == "__main__":

  with open("config/dr.yaml", "r") as file:
    config_dict = yaml.safe_load(file)
  main(config_dict)
