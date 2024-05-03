import itertools
import math
from collections import namedtuple
from typing import Optional, Sequence

import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from .autoregressive import Autoregressive
from .conditional import (
  ConditionalChain,
  ConditionalInverse,
  ConditionalTransformed,
)


def make_conditioner(
  event_shape: Sequence[int],
  hidden_sizes: Sequence[int],
  num_bijector_params: int,
  periodized: bool = False,
  init_to_identity: bool = True,
) -> hk.Sequential:
  """Creates an MLP conditioner for each layer of the flow."""

  def conditioner(x: Optional[Array] = None, name: str = ""):
    if x is None or x.shape[-1] == 0:
      init = jnp.zeros if init_to_identity else hk.initializers.RandomNormal(
        stddev=1. / math.sqrt(num_bijector_params)
      )
      return hk.get_parameter(
        "first",
        shape=(np.prod(event_shape), num_bijector_params),
        init=init,
      )

    x = hk.Flatten(preserve_dims=-len(event_shape))(x)
    if periodized:
      x = jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1)
    x = hk.nets.MLP(hidden_sizes, activate_final=True, name=f"mlp_{name}")(x)
    # We initialize this linear layer to zero so that the flow is initialized
    # to the identity function.
    out_kwargs = dict(
      w_init=jnp.zeros,
      b_init=jnp.zeros,
    ) if init_to_identity else dict()
    x = hk.Linear(
      np.prod(event_shape) * num_bijector_params,
      name=f"linear_out_{name}",
      **out_kwargs
    )(x)
    x = hk.Reshape(
      tuple(event_shape) + (num_bijector_params, ), preserve_dims=-1
    )(x)

    return x

  return conditioner


def make_flow_model(
  event_shape: Sequence[int],
  num_layers: int,
  hidden_sizes: Sequence[int],
  num_bins: int,
  periodized: bool = False,
  init_to_identity: bool = True,
) -> distrax.Transformed:
  """Creates a flow model supported on [0,1].

  If periodized is True, the flow is supported on the torus, which is the
  cartesian product of `len(event_shape)` circles. This is achieved by the
  boundary conditions: f(0)=0, f(2*pi)=2*pi, df(x)>0, df(0)=df(2*pi).

  For RQS, this is achieved by setting the first and the last slope the same,
  since it is already monotonic.

  Args:
  """
  event_dim = np.prod(event_shape)

  def bijector_fn(params: Array):

    return distrax.RationalQuadraticSpline(
      params,
      range_min=0. if periodized else -10.,
      range_max=2 * np.pi if periodized else 10.,
      # TODO: the tori paper uses 1e-3, check whether the default 1e-4 is better
      min_knot_slope=1e-4,
      boundary_slopes='circular' if periodized else 'unconstrained'
    )

  # Number of parameters for the rational-quadratic spline:
  # - `num_bins` bin widths
  # - `num_bins` bin heights
  # - `num_bins + 1` knot slopes
  # for a total of `3 * num_bins + 1` parameters.
  num_bijector_params = 3 * num_bins + 1

  layers = []

  if True:  # autoregressive
    # the conditioner has event shape of 1 since autoregressive
    # decomposition is used
    perms = itertools.cycle(itertools.permutations(range(event_dim)))
    for l in range(num_layers):  # stacking produce weird artefacts
      layer = Autoregressive(
        bijector=bijector_fn,
        conditioner=make_conditioner(
          (1, ), hidden_sizes, num_bijector_params, periodized, init_to_identity
        ),
        event_shape=event_shape,
        cond_shape=(1, ),  # TODO: remove hardcoding here
        permutation=next(perms),
      )
      layers.append(layer)

  else:
    if periodized:
      first_layer_params = hk.get_parameter(
        "first",
        shape=(np.prod(event_shape), num_bijector_params),
        init=hk.initializers.RandomNormal(
          stddev=1. / math.sqrt(num_bijector_params)
        ),
      )
      layers.append(distrax.Block(bijector_fn(first_layer_params), 1))

    if periodized:
      # override num_layers
      # TODO: why when > 1 it converges to uniform?
      # num_layers = event_dim - 1
      mask = jnp.zeros(event_dim).astype(bool)
      # mask = mask.at[0].set(True)
    else:
      # Alternating binary mask.
      mask = jnp.arange(0, event_dim) % 2
      mask = jnp.reshape(mask, event_shape)
      mask = mask.astype(bool)

    mask = jnp.arange(0, event_dim) % 2
    mask = jnp.reshape(mask, event_shape)
    mask = mask.astype(bool)

    # use x to condition y and z
    for l in range(num_layers):
      if periodized:
        mask = jnp.eye(event_dim)[l % event_dim].astype(bool)
      layer = distrax.MaskedCoupling(
        mask=mask,
        bijector=bijector_fn,
        conditioner=make_conditioner(
          event_shape, hidden_sizes, num_bijector_params, periodized
        )
      )
      layers.append(layer)

      if not periodized:
        # Flip the mask after each layer.
        mask = jnp.logical_not(mask)

  # We invert the flow so that the `forward` method is called with `log_prob`.
  flow = ConditionalInverse(ConditionalChain(layers))
  base_distribution = distrax.Independent(
    # distrax.Uniform(
    #   low=jnp.zeros(event_shape),
    #   high=jnp.ones(event_shape) * (2 * np.pi if periodized else 1.)
    # ),
    distrax.Normal(
      loc=jnp.zeros(event_shape),
      scale=jnp.ones(event_shape)
    ),
    reinterpreted_batch_ndims=len(event_shape)
  )

  return ConditionalTransformed(base_distribution, flow)


def RQSFlow(
  event_shape: Sequence[int],
  num_layers: int,
  hidden_sizes: Sequence[int],
  num_bins: int,
  periodized: bool = False,
):

  def model() -> Array:
    flow = make_flow_model(
      event_shape=event_shape,
      num_layers=num_layers,
      hidden_sizes=hidden_sizes,
      num_bins=num_bins,
      periodized=periodized,
    )

    # NOTE: in the context of DPW, x is the parameter space, also written as xi
    # and y is the physical space, also written as r
    def forward_jac(x, c):
      return jax.vmap(jax.jacfwd(flow.bijector.forward))(x, c)

    def inverse_jac(y, c):
      return jax.vmap(jax.jacfwd(flow.bijector.inverse))(y, c)

    def gauge_potential(x, c):
      return jax.jacfwd(lambda x_: flow.bijector.forward_and_log_det(x_, c)[1]
                        )(x)

    return (
      # lambda r: flow.bijector.forward(flow.bijector.inverse(r)),
      flow.log_prob,
      namedtuple(
        "Flow", [
          "log_prob", "sample", "sample_and_log_prob", "forward", "inverse",
          "forward_jac", "inverse_jac", "gauge_potential"
        ]
      )(
        flow.log_prob, flow.sample, flow.sample_and_log_prob,
        flow.bijector.forward, flow.bijector.inverse, forward_jac, inverse_jac,
        gauge_potential
      ),
    )

  return model