"""Benchmark initialized VCNF sampling and log-density evaluation."""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp

from cnf_ot.models.flows import RQSFlow


jax.config.update("jax_enable_x64", True)


def synchronize(value):
  """Block until all arrays in a JAX result are ready."""
  return jax.tree.map(lambda x: x.block_until_ready(), value)


def timed_runs(fn, args, repeats):
  durations_ms = []
  for _ in range(repeats):
    start = time.perf_counter_ns()
    synchronize(fn(*args))
    end = time.perf_counter_ns()
    durations_ms.append((end - start) / 1e6)
  return durations_ms


def benchmark_dimension(dim, batch_size, warmups, repeats, seed):
  model = RQSFlow(
    event_shape=(dim,),
    num_layers=2,
    hidden_sizes=[16, 16],
    num_bins=5,
    periodized=False,
  )
  model = hk.without_apply_rng(hk.multi_transform(model))

  key = jax.random.PRNGKey(seed)
  init_key, sample_key, data_key = jax.random.split(key, 3)
  params = model.init(
    init_key,
    jnp.zeros((1, dim), dtype=jnp.float64),
    jnp.zeros((1,), dtype=jnp.float64),
  )
  parameter_count = sum(x.size for x in jax.tree.leaves(params))

  sample_fn = jax.jit(model.apply.sample, static_argnames=("sample_shape",))
  log_prob_fn = jax.jit(model.apply.log_prob)
  sample_cond = jnp.full((batch_size, 1), 0.5, dtype=jnp.float64)
  log_prob_cond = jnp.full((1,), 0.5, dtype=jnp.float64)
  data = jax.random.normal(data_key, (batch_size, dim), dtype=jnp.float64)

  sample_args = (params, sample_key)
  sample_kwargs = {"sample_shape": (batch_size,), "cond": sample_cond}

  def sample_call(params_, key_):
    return sample_fn(params_, seed=key_, **sample_kwargs)

  log_prob_args = (params, data, log_prob_cond)

  synchronize(sample_call(*sample_args))
  synchronize(log_prob_fn(*log_prob_args))
  for _ in range(warmups):
    synchronize(sample_call(*sample_args))
    synchronize(log_prob_fn(*log_prob_args))

  sample_ms = timed_runs(sample_call, sample_args, repeats)
  log_prob_ms = timed_runs(log_prob_fn, log_prob_args, repeats)

  return {
    "dimension": dim,
    "parameters": parameter_count,
    "batch_size": batch_size,
    "sampling_ms": {
      "median": statistics.median(sample_ms),
      "min": min(sample_ms),
      "max": max(sample_ms),
      "all": sample_ms,
    },
    "log_density_ms": {
      "median": statistics.median(log_prob_ms),
      "min": min(log_prob_ms),
      "max": max(log_prob_ms),
      "all": log_prob_ms,
    },
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--dimensions", type=int, nargs="+", default=[2, 10, 20])
  parser.add_argument("--batch-size", type=int, default=10_000)
  parser.add_argument("--warmups", type=int, default=5)
  parser.add_argument("--repeats", type=int, default=30)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument(
    "--output", type=Path, default=Path("benchmark_results/inference.json")
  )
  args = parser.parse_args()

  results = {
    "platform": platform.platform(),
    "python": platform.python_version(),
    "jax": jax.__version__,
    "devices": [str(device) for device in jax.devices()],
    "precision": "float64",
    "flow": {
      "layers": 2,
      "conditioner_hidden_sizes": [16, 16],
      "spline_bins": 5,
    },
    "warmups": args.warmups,
    "repeats": args.repeats,
    "results": [
      benchmark_dimension(
        dim, args.batch_size, args.warmups, args.repeats, args.seed
      )
      for dim in args.dimensions
    ],
  }

  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(results, indent=2) + "\n")
  print(json.dumps(results, indent=2))


if __name__ == "__main__":
  main()
