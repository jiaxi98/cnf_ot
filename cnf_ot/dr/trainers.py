from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import ml_collections
import optax
from box import Box
from jaxtyping import Array
from matplotlib import pyplot as plt
from tqdm import tqdm

from cnf_ot.models.flows import RQSFlow
from cnf_ot.types import OptState, PRNGKey


def train(
  rng: PRNGKey,
  data: jnp.ndarray,
  dim: int,
  sub_dim: int,
  model: str,
  epochs: int,
  config_dict: ml_collections.ConfigDict
):
  """Train a NF model to learn the coordinate transformation.
  
  The encoder and decoder are defined in the ambient space :math:`\mathbb{R}^n`
  and their mapping are

  .. math::

    (x_1, x_2, ..., x_n)  \overset{f_{\theta}^{\text{encoder}}}{\mapsto} 
    (y_1, y_2, ..., y_n),    \\
    (y_1, y_2, ..., y_d, 0, ..., 0)  \overset{f_{\theta}^{\text{decoder}}}{\mapsto}
    (x_1', x_2', ..., x_n'), \\
    L(\theta) = \mathbb{E}_{\mathbf{x}} \left\| \mathbf{x} - \mathbf{x}' \right\|^2
  
  where :math:`\mathbf{x}` is the input data, :math:`\mathbf{x}'` is the
  reconstructed data, :math:`\theta` is the parameters of the model, and
  :math:`L(\theta)` is the loss function.
  """
  
  config = Box(config_dict)
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
    params = {
      "encoder": encoder.init(encoder_rng, jnp.zeros((1, dim))),
      "decoder": decoder.init(decoder_rng, jnp.zeros((1, dim)))
    }
  elif model == "dec_only":
    # decoder only architecture
    decoder_rng, rng = jax.random.split(rng)
    decoder_inverse_fn = jax.jit(decoder.apply.inverse)
    params = decoder.init(decoder_rng, jnp.zeros((1, dim)))
  schedule = optax.piecewise_constant_schedule(
    init_value=config.train.lr,
    boundaries_and_scales={int(b): 0.1
                           for b in jnp.arange(5000, epochs, 5000)}
  )
  optimizer = optax.adam(schedule)
  opt_state = optimizer.init(params)

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
    loss, grads = jax.value_and_grad(loss_fn)(params, data)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, new_opt_state

  loss_hist = []
  iters = tqdm(range(epochs))
  # param_count = sum(x.size for x in jax.tree.leaves(params))
  # print("Network parameters: {}".format(param_count))
  for _ in iters:
    # samples = generate_low_dim_data(key, dim, config.type, batch_size)
    loss, params, opt_state = update(params, opt_state)
    loss_hist.append(loss)
    lr = schedule(_)
    desc_str = f"{lr:.2e}|{loss=:.4e}"
    iters.set_description_str(desc_str)

  plt.plot(loss_hist)
  plt.yscale("log")
  plt.savefig(f"results/fig/loss_{model}.png")
  plt.clf()
  if model == "enc_dec":
    return encoder, decoder, params, loss_hist
  elif model == "dec_only":
    return decoder, params, loss_hist


def static_path_finder(
  config_dict: ml_collections.ConfigDict, data: jnp.ndarray,
):

  config = Box(config_dict)
  dim = config.dim
  model = config.model
  rng = jax.random.PRNGKey(config.seed)
  epochs = config.train.epochs
  sub_dim = int(config.type[1])

  data1 = data[jnp.linalg.norm(data - start[None], axis=-1) < r]
  data2 = data[jnp.linalg.norm(data - end[None], axis=-1) < r]
  overlap = data2[jnp.linalg.norm(data2 - start[None], axis=-1) < r]
  print(f"data: {data.shape[0]}; data1: {data1.shape[0]};\
    data2: {data2.shape[0]}; overlap: {overlap.shape[0]}")
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
    return encoders, decoders, params
  elif model == "dec_only":
    decoder1, params1, _ = train(
      rng, data1, dim, sub_dim, model, epochs, config
    )
    decoder2, params2, _ = train(
      rng, data2, dim, sub_dim, model, epochs, config
    )
    decoders = [decoder1, decoder2]
    params = [params1, params2]
    return decoders, params
  

def dynamics_path_finder(
  config_dict: ml_collections.ConfigDict, data: jnp.ndarray,
  start: jnp.ndarray, end: jnp.ndarray, init_r: float = 3, relax: float = 1.5
):
  """Find a path between two points dynamically.
  
  Use greedy search to find the charts.
  TODO: add reference in our future paper

  Args:
    config_dict: config file.
    data: data lying on the manifold.
    start: start point
    end: end point
    init_r: initial radius for the chart
    relax: relaxation factor for the radius
  Returns:
    charts: list of charts in the order of transition
    pos: list of centers of the charts, :py:func:`pos[-1] = end`
    radius: list of radii of the charts.
    encoders: list of encoders.
    decoders: list of decoders.
    params: list of parameters for encoders and decoders.
  """
  
  config = Box(config_dict)
  dim = config.dim
  model = config.model
  rng = jax.random.PRNGKey(config.seed)
  epochs = config.train.epochs
  sub_dim = int(config.type[1])

  charts = []
  pos = []
  radius = []
  encoders = []
  decoders = []
  params = []
  pos_ = start
  index = 0

  while True:
    print(f"Finding {index}th chart...")
    r = init_r
    while True:
      chart = data[jnp.linalg.norm(data - pos_, axis=-1) < r]
      encoder, decoder, params_, loss = train(
        rng, chart, dim, sub_dim, model, epochs, config
      )
      if loss[-1] < 1e-1:
        break
      r /= relax
    charts.append(chart)
    pos.append(pos_)
    radius.append(r)
    encoders.append(encoder)
    decoders.append(decoder)
    params.append(params_)
    print(f"Chart {index} found radius {r:.2f} with loss {loss[-1]}")
    if jnp.linalg.norm(pos_ - end) < r:
      print(f"Chart {index} is close to end point")
      break
    pos_ = chart[jnp.argmin(jnp.linalg.norm(chart - end, axis=-1))]
    index += 1
    print(f"L2 dist between current pos and end: {jnp.linalg.norm(pos_-end):.3f}")
  pos.append(end)

  return charts, pos, radius, encoders, decoders, params
