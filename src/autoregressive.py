from typing import Any, Callable, Optional, Sequence, Tuple

import jax.numpy as jnp
from distrax._src.bijectors import bijector as base
from distrax._src.utils import conversion

Array = base.Array
BijectorParams = Any


class Autoregressive(base.Bijector):
  """Auto-regressive coupling bijector"""

  def __init__(
    self,
    conditioner: Callable[[Array], BijectorParams],
    bijector: Callable[[BijectorParams], base.BijectorLike],
    event_shape: int = 1,
    permutation: Optional[Sequence[int]] = None,
  ):
    self._conditioner = conditioner
    self._bijector = bijector
    self._event_shape = event_shape
    self._event_ndims = 1  # autoregressive
    self.permutation = permutation or list(range(sum(event_shape)))
    super().__init__(event_ndims_in=self._event_ndims)

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

  def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
    """Computes y = f(x) and log|det J(f)(x)|."""
    self._check_forward_input_shape(x)

    y = jnp.zeros_like(x)
    logdet = jnp.zeros_like(x)
    ndims = x.shape[-1]
    for d in range(ndims):
      i = self.permutation[d]
      if i == 0:
        params_i = self._conditioner(None)
      else:
        params_i = self._conditioner(x[..., self.permutation[:d]], d)
      x_i = x[..., i:i + 1]
      y_i, logdet_i = self._inner_bijector(params_i).forward_and_log_det(x_i)
      y = y.at[..., i:i + 1].set(y_i)
      # logdet += logdet_i
      logdet = logdet.at[..., i:i + 1].set(logdet_i)

    logdet = logdet.sum(-1)

    return y, logdet

  def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
    self._check_inverse_input_shape(y)

    x = jnp.zeros_like(y)
    logdet = jnp.zeros_like(y)
    ndims = y.shape[-1]
    for d in range(ndims):
      i = self.permutation[d]
      if i == 0:
        params_i = self._conditioner(None)
      else:
        params_i = self._conditioner(x[..., self.permutation[:d]], d)
      y_i = y[..., i:i + 1]
      x_i, logdet_i = self._inner_bijector(params_i).inverse_and_log_det(y_i)
      x = x.at[..., i:i + 1].set(x_i)
      logdet = logdet.at[..., i:i + 1].set(logdet_i)

    logdet = logdet.sum(-1)

    return x, logdet
