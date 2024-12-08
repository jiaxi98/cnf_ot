from typing import Any, Mapping

import jax
import numpy as np

PRNGKey = jax.random.PRNGKey
Batch = Mapping[str, np.ndarray]
OptState = Any
