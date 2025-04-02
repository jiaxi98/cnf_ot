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
"""
Extend distrax to support conditional normalizing flow

NOTE: The only difference between this file and the one in current source code
is that one set the cond argument to have default value of None, which
is used for unconditional normalizing flows.
"""
import abc
import functools
import operator
from typing import List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from distrax._src.bijectors import bijector as base
from distrax._src.distributions import distribution as dist_base
from distrax._src.distributions.transformed import Transformed
from distrax._src.utils import conversion

Array = base.Array
BijectorLike = base.BijectorLike
BijectorT = base.BijectorT
PRNGKey = dist_base.PRNGKey
Array = dist_base.Array
EventT = dist_base.EventT
ShapeT = dist_base.ShapeT
IntLike = dist_base.IntLike


class ConditionalBijector(base.Bijector):
  """A map `f(x,c)=y` bijective in `x` only, whose inverse is defined as
  `f^{-1}(y,c)=x`.

  Typically, a bijector subclass will implement the following methods:

  - `forward_and_log_det(x, c)` (required)
  - `inverse_and_log_det(y, c)` (optional)
  """

  def __init__(
    self,
    cond_shape: ShapeT,
    event_ndims_in: int,
    event_ndims_out: Optional[int] = None,
    is_constant_jacobian: bool = False,
    is_constant_log_det: Optional[bool] = None
  ):
    """Initializes a Bijector.

    Args:
      cond_ndims_in: Number of input conditional dimensions.
    """
    self.cond_shape = cond_shape
    super().__init__(
      event_ndims_in=event_ndims_in,
      event_ndims_out=event_ndims_out,
      is_constant_jacobian=is_constant_jacobian,
      is_constant_log_det=is_constant_log_det,
    )

  def forward(self, x: Array, c: Array) -> Array:
    """Computes y = f(x,c)."""
    y, _ = self.forward_and_log_det(x, c)
    return y

  def inverse(self, y: Array, c: Array) -> Array:
    """Computes x = f^{-1}(y)."""
    x, _ = self.inverse_and_log_det(y, c)
    return x

  def forward_log_det_jacobian(self, x: Array, c: Array) -> Array:
    """Computes log|det J(f)(x, c)|."""
    _, logdet = self.forward_and_log_det(x, c)
    return logdet

  def inverse_log_det_jacobian(self, y: Array, c: Array) -> Array:
    """Computes log|det J(f^{-1})(y, c)|."""
    _, logdet = self.inverse_and_log_det(y, c)
    return logdet

  @abc.abstractmethod
  def forward_and_log_det(self, x: Array, c: Array) -> Tuple[Array, Array]:
    """Computes y = f(x,c) and log|det J(f)(x,c)|."""

  def inverse_and_log_det(self, y: Array, c: Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y,c) and log|det J(f^{-1})(y,c)|."""
    raise NotImplementedError(
      f"Bijector {self.name} does not implement `inverse_and_log_det`."
    )


class ConditionalChain(ConditionalBijector):

  def __init__(self, bijectors: Sequence[BijectorLike]):
    """Initializes a Chain bijector.

    Args:
      bijectors: a sequence of bijectors to be composed into one. Each bijector
        can be a distrax bijector, a TFP bijector, or a callable to be wrapped
        by `Lambda`. The sequence must contain at least one bijector.
    """
    if not bijectors:
      raise ValueError("The sequence of bijectors cannot be empty.")
    self._bijectors = [
      b if isinstance(b, ConditionalBijector) else
      AsConditional(conversion.as_bijector(b)) for b in bijectors
    ]

    # Check that neighboring bijectors in the chain have compatible dimensions
    for i, (outer, inner) in enumerate(
      zip(self._bijectors[:-1], self._bijectors[1:])
    ):
      if outer.event_ndims_in != inner.event_ndims_out:
        raise ValueError(
          f"The chain of bijector event shapes are incompatible. Bijector "
          f"{i} ({outer.name}) expects events with {outer.event_ndims_in} "
          f"dimensions, while Bijector {i+1} ({inner.name}) produces events "
          f"with {inner.event_ndims_out} dimensions."
        )

    is_constant_jacobian = all(b.is_constant_jacobian for b in self._bijectors)
    is_constant_log_det = all(b.is_constant_log_det for b in self._bijectors)
    super().__init__(
      cond_shape=self._bijectors[-1].cond_shape,
      event_ndims_in=self._bijectors[-1].event_ndims_in,
      event_ndims_out=self._bijectors[0].event_ndims_out,
      is_constant_jacobian=is_constant_jacobian,
      is_constant_log_det=is_constant_log_det
    )

  @property
  def bijectors(self) -> List[BijectorT]:
    """The list of bijectors in the chain."""
    return self._bijectors

  def forward(self, x: Array, c: Array) -> Array:
    """Computes y = f(x,c)."""
    for bijector in reversed(self._bijectors):
      x = bijector.forward(x, c)
    return x

  def inverse(self, y: Array, c: Array) -> Array:
    """Computes x = f^{-1}(y,c)."""
    for bijector in self._bijectors:
      y = bijector.inverse(y, c)
    return y

  def forward_and_log_det(self, x: Array, c: Array) -> Tuple[Array, Array]:
    """Computes y = f(x,c) and log|det J(f)(x,c)|."""
    x, log_det = self._bijectors[-1].forward_and_log_det(x, c)
    for bijector in reversed(self._bijectors[:-1]):
      x, ld = bijector.forward_and_log_det(x, c)
      log_det += ld
    return x, log_det

  def inverse_and_log_det(self, y: Array, c: Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y) and log|det J(f^{-1})(y,c)|."""
    y, log_det = self._bijectors[0].inverse_and_log_det(y, c)
    for bijector in self._bijectors[1:]:
      y, ld = bijector.inverse_and_log_det(y, c)
      log_det += ld
    return y, log_det

  def same_as(self, other: base.Bijector) -> bool:
    """Returns True if this bijector is guaranteed to be the same as `other`."""
    if type(other) is ConditionalChain:  # pylint: disable=unidiomatic-typecheck
      if len(self.bijectors) != len(other.bijectors):
        return False
      for bij1, bij2 in zip(self.bijectors, other.bijectors):
        if not bij1.same_as(bij2):
          return False
      return True
    elif len(self.bijectors) == 1:
      return self.bijectors[0].same_as(other)

    return False


class ConditionalInverse(ConditionalBijector):

  def __init__(self, bijector: BijectorLike):
    """Initializes an Inverse bijector.

    Args:
      bijector: the bijector to be inverted. It can be a distrax bijector, a TFP
        bijector, or a callable to be wrapped by `Lambda`.
    """
    self._bijector = conversion.as_bijector(bijector)
    super().__init__(
      cond_shape=self._bijector.cond_shape,
      event_ndims_in=self._bijector.event_ndims_out,
      event_ndims_out=self._bijector.event_ndims_in,
      is_constant_jacobian=self._bijector.is_constant_jacobian,
      is_constant_log_det=self._bijector.is_constant_log_det
    )

  @property
  def bijector(self) -> ConditionalBijector:
    """The base bijector that was the input to `Inverse`."""
    return self._bijector

  def forward(self, x: Array, c: Array) -> Array:
    """Computes y = f(x)."""
    return self._bijector.inverse(x, c)

  def inverse(self, y: Array, c: Array) -> Array:
    """Computes x = f^{-1}(y)."""
    return self._bijector.forward(y, c)

  def forward_log_det_jacobian(self, x: Array, c: Array) -> Array:
    """Computes log|det J(f)(x)|."""
    return self._bijector.inverse_log_det_jacobian(x, c)

  def inverse_log_det_jacobian(self, y: Array, c: Array) -> Array:
    """Computes log|det J(f^{-1})(y)|."""
    return self._bijector.forward_log_det_jacobian(y, c)

  def forward_and_log_det(self, x: Array, c: Array) -> Tuple[Array, Array]:
    """Computes y = f(x) and log|det J(f)(x)|."""
    return self._bijector.inverse_and_log_det(x, c)

  def inverse_and_log_det(self, y: Array, c: Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
    return self._bijector.forward_and_log_det(y, c)

  def same_as(self, other: base.Bijector) -> bool:
    """Returns True if this bijector is guaranteed to be the same as `other`."""
    if type(other) is ConditionalInverse:  # pylint: disable=unidiomatic-typecheck
      return self.bijector.same_as(other.bijector)
    return False


class AsConditional(ConditionalBijector):
  """Wrap around a normal bijector and make it trivally conditional"""

  def __init__(self, bijector: BijectorLike):
    self._bijector = conversion.as_bijector(bijector)
    super().__init__(
      cond_shape=(0, ),
      event_ndims_in=self._bijector.event_ndims_in,
      event_ndims_out=self._bijector.event_ndims_out,
      is_constant_jacobian=self._bijector.is_constant_jacobian,
      is_constant_log_det=self._bijector.is_constant_log_det,
    )

  def forward_and_log_det(self, x: Array, c: Array) -> Tuple[Array, Array]:
    """Computes y = f(x,c) and log|det J(f)(x,c)|."""
    return self._bijector.forward_and_log_det(x)

  def inverse_and_log_det(self, y: Array, c: Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y,c) and log|det J(f^{-1})(y,c)|."""
    return self._bijector.inverse_and_log_det(y)


class ConditionalTransformed(Transformed):
  """A conditional distribution `p(Y|C)` obtained from transforming a continous
  R.V. `X` from the base distribution `p(X)` by a conditional bijector `f(X,C)`
  bijective
  in `X`, whose inverse is defined as `f^{-1}(Y,C)=X`.

  The probability density of `Y` can be computed by:

  `log p(y|c) = log p(x) - log|det J(f)(x, c)|`

  where `J(f)(x, c)` is the Jacobian matrix of `f` evaluated at `x = f^{-1}(y, c)`.
  """

  def _infer_shapes_and_dtype(self):
    """Infer the batch shape, event shape, and dtype by tracing `forward`."""
    dummy_shape = self.distribution.batch_shape + self.distribution.event_shape
    dummy = jnp.zeros(dummy_shape, dtype=self.distribution.dtype)
    # TODO: change this
    assert hasattr(self.bijector, "cond_shape")
    dummy_cond_shape = self.bijector.cond_shape
    dummy_cond = jnp.zeros(dummy_cond_shape, dtype=self.distribution.dtype)
    shape_dtype = jax.eval_shape(self.bijector.forward, dummy, dummy_cond)
    self._cond_shape = self.bijector.cond_shape

    self._dtype = shape_dtype.dtype

    if self.bijector.event_ndims_out == 0:
      self._event_shape = ()
      self._batch_shape = shape_dtype.shape
    else:
      # pylint: disable-next=invalid-unary-operand-type
      self._event_shape = shape_dtype.shape[-self.bijector.event_ndims_out:]
      # pylint: disable-next=invalid-unary-operand-type
      self._batch_shape = shape_dtype.shape[:-self.bijector.event_ndims_out]

  def log_prob(self, value: EventT, cond: Array) -> Array:
    """See `Distribution.log_prob`."""
    x, ildj_y = self.bijector.inverse_and_log_det(value, cond)
    lp_x = self.distribution.log_prob(x)
    lp_y = lp_x + ildj_y
    return lp_y

  def sample(
    self,
    *,
    cond: Array,
    seed: Union[IntLike, PRNGKey],
    sample_shape: Union[IntLike, Sequence[IntLike]] = ()
  ) -> EventT:
    """Samples an event.

    Args:
      cond: condition
      seed: PRNG key or integer seed.
      sample_shape: Additional leading dimensions for sample.

    Returns:
      A sample of shape `sample_shape + self.batch_shape + self.event_shape`.
    """
    rng, sample_shape = dist_base.convert_seed_and_sample_shape(
      seed, sample_shape
    )
    num_samples = functools.reduce(operator.mul, sample_shape, 1)  # product

    samples = self._sample_n(rng, num_samples, cond)
    return jax.tree_util.tree_map(
      lambda t: t.reshape(sample_shape + t.shape[1:]), samples
    )

  def sample_and_log_prob(
    self,
    *,
    cond: Array,
    seed: Union[IntLike, PRNGKey],
    sample_shape: Union[IntLike, Sequence[IntLike]] = ()
  ) -> Tuple[EventT, Array]:
    """Returns a sample and associated log probability. See `sample`."""
    rng, sample_shape = dist_base.convert_seed_and_sample_shape(
      seed, sample_shape
    )
    num_samples = functools.reduce(operator.mul, sample_shape, 1)  # product

    samples, log_prob = self._sample_n_and_log_prob(rng, num_samples, cond)
    samples, log_prob = jax.tree_util.tree_map(
      lambda t: t.reshape(sample_shape + t.shape[1:]), (samples, log_prob)
    )
    return samples, log_prob

  def _sample_n(self, rng: PRNGKey, n: int, cond: Array) -> Array:
    """Returns `n` samples."""
    x = self.distribution.sample(seed=rng, sample_shape=n)
    y = jax.vmap(self.bijector.forward)(x, cond)
    return y

  def _sample_n_and_log_prob(self, rng: PRNGKey, n: int,
                             cond: Array) -> Tuple[Array, Array]:
    """Returns `n` samples and their log probs.

    This function is more efficient than calling `sample` and `log_prob`
    separately, because it uses only the forward methods of the bijector. It
    also works for bijectors that don't implement inverse methods.

    Args:
      rng: PRNG key.
      n: Number of samples to generate.

    Returns:
      A tuple of `n` samples and their log probs.
    """
    x, lp_x = self.distribution.sample_and_log_prob(seed=rng, sample_shape=n)
    y, fldj = jax.vmap(self.bijector.forward_and_log_det)(x, cond)
    lp_y = jax.vmap(jnp.subtract)(lp_x, fldj)
    return y, lp_y

  def mean(self, cond: Array) -> Array:
    """Calculates the mean."""
    if self.bijector.is_constant_jacobian:
      return self.bijector.forward(self.distribution.mean(), cond)
    else:
      raise NotImplementedError(
        "`mean` is not implemented for this transformed distribution, "
        "because its bijector's Jacobian is not known to be constant."
      )

  def mode(self, cond: Array) -> Array:
    """Calculates the mode."""
    if self.bijector.is_constant_log_det:
      return self.bijector.forward(self.distribution.mode(), cond)
    else:
      raise NotImplementedError(
        "`mode` is not implemented for this transformed distribution, "
        "because its bijector's Jacobian determinant is not known to be "
        "constant."
      )

  def entropy(  # pylint: disable=arguments-differ
      self,
      cond: Array,
      input_hint: Optional[Array] = None) -> Array:
    """Calculates the Shannon entropy (in Nats).

    Only works for bijectors with constant Jacobian determinant.

    Args:
      input_hint: an example sample from the base distribution, used to compute
        the constant forward log-determinant. If not specified, it is computed
        using a zero array of the shape and dtype of a sample from the base
        distribution.

    Returns:
      the entropy of the distribution.

    Raises:
      NotImplementedError: if bijector's Jacobian determinant is not known to be
                           constant.
    """
    if self.bijector.is_constant_log_det:
      if input_hint is None:
        shape = self.distribution.batch_shape + self.distribution.event_shape
        input_hint = jnp.zeros(shape, dtype=self.distribution.dtype)
      entropy = self.distribution.entropy()
      fldj = self.bijector.forward_log_det_jacobian(input_hint, cond)
      return entropy + fldj
    else:
      raise NotImplementedError(
        "`entropy` is not implemented for this transformed distribution, "
        "because its bijector's Jacobian determinant is not known to be "
        "constant."
      )
