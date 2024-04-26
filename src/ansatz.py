import dataclasses
from collections import OrderedDict, namedtuple
from typing import Optional, Sequence

import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex, Float, Int

from src.flows import make_flow_model


@dataclasses.dataclass
class NF:
  r"""Normalizing flow

  Contains:
    1. flow-based density ansatz

  The latent space is denoted as z whereas the data space is denoted as r
  """
  hidden_sizes: Sequence[int] = (64, 64)
  """hidden sizes for the inner bijector of the RQS flow"""
  num_bins: int = 3
  """number of spline points of the RQS flow"""

  init_to_identity: bool = True

  dim: int = 3
  """dimension of the physical space."""

  def __post_init__(self):
    event_shape = (self.dim, )
    self.flow = make_flow_model(
      event_shape=event_shape,
      num_layers=1,  # only 1 layer works for now
      hidden_sizes=self.hidden_sizes,
      num_bins=self.num_bins,
      periodized=True,
      init_to_identity=self.init_to_identity,
    )

    affine = distrax.Inverse(
      distrax.UnconstrainedAffine(matrix=np.eye(self.dim), bias=np.zeros(self.dim))
    )
    base_distribution = distrax.Independent(
      distrax.Uniform(
        low=jnp.zeros(event_shape), high=jnp.ones(event_shape) * 2 * np.pi
      ),
      reinterpreted_batch_ndims=len(event_shape)
    )
    self.uniform_sampler = distrax.Transformed(base_distribution, affine)

  def init(self, r: Array):
    """initialize all the learnable parameters
    TODO: should this be r or z? what is it meaning?"""
    return self.flow.log_prob(r)

  def forward_jac_det(self, z: Float[Array, "3"]) -> Float[Array, ""]:
    return jnp.exp(self.flow.bijector.forward_and_log_det(z)[1])

  def sample(self, seed, sample_shape):
    return self.flow.sample(seed=seed, sample_shape=sample_shape)

  def sample_uniform(self, seed, sample_shape):
    return self.uniform_sampler.sample(seed=seed, sample_shape=sample_shape)

  def sample_and_log_prob(self, seed, sample_shape):
    return self.flow.sample_and_log_prob(seed=seed, sample_shape=sample_shape)

  def init_for_multitransform(self):
    api = OrderedDict(
      log_prob=self.flow.log_prob,
      sample=self.sample,
      sample_uniform=self.sample_uniform,
      sample_and_log_prob=self.sample_and_log_prob,
      forward=self.flow.bijector.forward,
      forward_and_log_det=self.flow.bijector.forward_and_log_det,
      forward_jac_det=self.forward_jac_det,
      inverse=self.flow.bijector.inverse,
      forward_jac=jax.jacfwd(self.flow.bijector.forward),
      inverse_jac=jax.jacfwd(self.flow.bijector.inverse),
    )

    return self.init, namedtuple("NF", list(api.keys()))(*api.values())
