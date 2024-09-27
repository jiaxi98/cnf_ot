# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Optional, Sequence, Tuple

import jax.numpy as jnp
from distrax._src.bijectors import bijector as base
from distrax._src.distributions import distribution as dist_base
from distrax._src.utils import conversion

from .conditional import ConditionalBijector

Array = base.Array
BijectorParams = Any

ShapeT = dist_base.ShapeT


class Autoregressive(ConditionalBijector):
  """Auto-regressive coupling bijector that can be optionally conditioned.

  Given a vector input x, decompose the distribution `p(x)` as
  `p(x_{i_1})p(x_{i_1}|p(x_{i_2}))p(x_{i_2}|p(x_{i_3}))...`
  where the ordering `i_n` is determined by the permutation.

  If conditional, the decomposition becomes
  `p(x_{i_1}|c)p(x_{i_1}|p(x_{i_2}))p(x_{i_2}|p(x_{i_3}))...`
  where `c` is the condition.
  """

  def __init__(
    self,
    conditioner: Callable[[Array], BijectorParams],
    bijector: Callable[[BijectorParams], base.BijectorLike],
    event_shape: ShapeT = (1, ),
    cond_shape: ShapeT = (0, ),
    permutation: Optional[Sequence[int]] = None,
    name: str = "",
  ):
    self._conditioner = conditioner
    self._bijector = bijector
    self._event_shape = event_shape
    self._event_ndims = 1  # autoregressive
    self.permutation = permutation if permutation is not None else list(range(sum(event_shape)))
    self.is_conditional = sum(cond_shape) > 0
    self._name = name
    super().__init__(event_ndims_in=self._event_ndims, cond_shape=cond_shape)

  @property
  def bijector(self) -> Callable[[BijectorParams], base.BijectorLike]:
    """The callable that returns the inner bijector."""
    return self._bijector

  @property
  def conditioner(self) -> Callable[[Array], BijectorParams]:
    """The conditioner function."""
    return self._conditioner

  def _inner_bijector(self, params: BijectorParams) -> base.Bijector:
    bijector = conversion.as_bijector(self._bijector(params))
    return bijector

  def forward_and_log_det(self, x: Array, c: Array) -> Tuple[Array, Array]:
    """Computes y = f(x,c) and log|det J(f)(x,c)|."""
    self._check_forward_input_shape(x)

    y = jnp.zeros_like(x)
    logdet = jnp.zeros_like(x)
    ndims = x.shape[-1]
    for d in range(ndims):
      i = self.permutation[d]
      layer_name = f"{self._name}_d{d}"
      if d == 0:
        # params_i = self._conditioner(
        #   c if self.is_conditional else None, layer_name
        # )
        params_i = self._conditioner(None, layer_name)
      else:
        cond_input = y[..., self.permutation[:d]]
        if self.is_conditional:
          c_ = jnp.broadcast_to(c, cond_input.shape[:-1] + c.shape)
          cond_input = jnp.concatenate([c_, cond_input], axis=-1)
        params_i = self._conditioner(cond_input, layer_name)
      x_i = x[..., i:i + 1]
      y_i, logdet_i = self._inner_bijector(params_i).forward_and_log_det(x_i)
      y = y.at[..., i:i + 1].set(y_i)
      # logdet += logdet_i
      logdet = logdet.at[..., i:i + 1].set(logdet_i)

    logdet = logdet.sum(-1)

    return y, logdet

  def inverse_and_log_det(self, y: Array, c: Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y,c) and log|det J(f^{-1})(y,c)|."""
    self._check_inverse_input_shape(y)

    x = jnp.zeros_like(y)
    logdet = jnp.zeros_like(y)
    ndims = y.shape[-1]
    for d in range(ndims):
      i = self.permutation[d]
      layer_name = f"{self._name}_d{d}"
      if d == 0:
        params_i = self._conditioner(None, layer_name)
      else:
        cond_input = y[..., self.permutation[:d]]
        if self.is_conditional:
          c_ = jnp.broadcast_to(c, cond_input.shape[:-1] + c.shape)
          cond_input = jnp.concatenate([c_, cond_input], axis=-1)
        params_i = self._conditioner(cond_input, layer_name)
      y_i = y[..., i:i + 1]
      x_i, logdet_i = self._inner_bijector(params_i).inverse_and_log_det(y_i)
      x = x.at[..., i:i + 1].set(x_i)
      logdet = logdet.at[..., i:i + 1].set(logdet_i)

    logdet = logdet.sum(-1)

    return x, logdet
