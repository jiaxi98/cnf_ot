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
from tqdm import tqdm

import cnf_ot.utils as utils
from cnf_ot.flows import RQSFlow
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
  rng = jax.random.PRNGKey(config.general.seed)
  epochs = config.train.epochs
  batch_size = config.train.batch_size
  model = RQSFlow(
    event_shape=(dim, ),
    num_layers=config.cnf.flow_num_layers,
    hidden_sizes=[config.cnf.hidden_size] * config.cnf.mlp_num_layers,
    num_bins=config.cnf.num_bins,
    periodized=False,
    cond_shape=(0, ),
  )
  model = hk.without_apply_rng(hk.multi_transform(model))
  forward_fn = jax.jit(model.apply.forward)
  model_rng, rng = jax.random.split(rng)
  # unconditional model
  params = model.init(model_rng, jnp.zeros((1, dim)))
  optimizer = optax.adam(config.train.lr)
  opt_state = optimizer.init(params)

  def generate_low_dim_data(
    key: PRNGKey, dim: int, type_: str, batch_size: int
  ):
    if type_ == "s1":
      # TODO: would be better to test it
      samples = jnp.zeros((batch_size, dim))
      samples = samples.at[:, :2].set(random.normal(key, (batch_size, 2)))
      samples /= jnp.sqrt(jnp.sum(samples**2, axis=-1))[:, None]
      return samples
    elif type_ == "s2":
      pass
    elif type_ == "t2":
      pass

  def loss_fn(params: hk.Params, x: jnp.ndarray) -> Array:
    y = forward_fn(params, x)
    return jnp.mean(jnp.sum(y[:, sub_dim:]**2, axis=-1)) +\
      jnp.mean((jnp.sum(y**2, axis=-1) - 1)**2)

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
  for step in iters:
    loss, params, opt_state = update(params, opt_state)
    loss_hist.append(loss)
    desc_str = f"{loss=:.4e}"
    iters.set_description_str(desc_str)

  utils.plot_dim_reduction(forward_fn, params, samples)


if __name__ == "__main__":

  with open("config/dr.yaml", "r") as file:
    config_dict = yaml.safe_load(file)
  main(config_dict)
