from typing import Any, Mapping, NamedTuple

import jax
import numpy as np
import optax

PRNGKey = jax.random.PRNGKey
Batch = Mapping[str, np.ndarray]
OptState = Any


class TrainingState(NamedTuple):
  params: Any
  opt_state: optax.OptState
  rng_key: jax.Array
  step: int


class Energies(NamedTuple):
  e_kin: float
  e_ext: float
  e_har: float
  e_xc: float
