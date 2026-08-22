"""Measure memory for one initialized VCNF training step."""

import argparse
import gc
import json
import os
import platform
import resource
import threading
import time
from functools import partial
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import optax
import psutil

from cnf_ot.mfc import applications
from cnf_ot.models.flows import RQSFlow


jax.config.update("jax_enable_x64", True)


def synchronize(value):
  return jax.tree.map(lambda x: x.block_until_ready(), value)


def rss_bytes():
  return psutil.Process().memory_info().rss


def fp_ou_flow_matching_loss(
  model, dim, a, sigma, dt, dx, params, cond, rng, batch_size
):
  sample_times = (cond - dt / 2, cond + dt / 2, cond)
  samples = [
    model.apply.sample(
      params,
      seed=rng,
      sample_shape=(batch_size,),
      cond=jnp.ones((batch_size, 1)) * sample_time,
    )
    for sample_time in sample_times
  ]
  velocity = (samples[1] - samples[0]) / dt

  score = jnp.zeros((batch_size, dim))
  for i in range(dim):
    offset = jnp.zeros((1, dim)).at[0, i].set(dx / 2)
    log_prob_plus = model.apply.log_prob(
      params, samples[2] + offset, cond=jnp.ones(1) * cond
    )
    log_prob_minus = model.apply.log_prob(
      params, samples[2] - offset, cond=jnp.ones(1) * cond
    )
    score = score.at[:, i].set((log_prob_plus - log_prob_minus) / dx)

  velocity += sigma * score
  truth = -a * samples[2]
  return jnp.mean((velocity - truth) ** 2) * dim / 2


def fp_ou_loss(
  model, dim, time_batch_size, params, rng, penalty, batch_size
):
  horizon = 1.0
  initial_beta = 4.0
  loss = penalty * applications.reverse_kl_loss_fn(
    model, dim, horizon, initial_beta, params, 0, rng, batch_size
  )
  times = jax.random.uniform(rng, (time_batch_size,)) * horizon
  for sample_time in times:
    loss += fp_ou_flow_matching_loss(
      model,
      dim,
      1.0,
      0.5,
      0.01,
      0.01,
      params,
      sample_time,
      rng,
      batch_size // 32,
    ) / time_batch_size * horizon
  return loss


def ot_gaussian_loss(
  model, dim, time_batch_size, params, rng, penalty, batch_size
):
  source_samples = (
    jax.random.normal(rng, (batch_size, dim), dtype=jnp.float64) - 3.0
  )
  target_samples = (
    jax.random.normal(rng, (batch_size, dim), dtype=jnp.float64) + 3.0
  )
  boundary_loss = -model.apply.log_prob(
    params, source_samples, cond=jnp.zeros((1,), dtype=jnp.float64)
  ).mean()
  boundary_loss -= model.apply.log_prob(
    params, target_samples, cond=jnp.ones((1,), dtype=jnp.float64)
  ).mean()

  loss = penalty * boundary_loss
  times = jax.random.uniform(rng, (time_batch_size,))
  for sample_time in times:
    loss += applications.kinetic_loss_fn(
      model,
      dim,
      0.01,
      params,
      sample_time,
      rng,
      batch_size // 32,
    ) / time_batch_size
  return loss


def make_loss(problem, model, dim, time_batch_size):
  if problem == "ot":
    return partial(ot_gaussian_loss, model, dim, time_batch_size)
  if problem == "rwpo":
    return partial(
      applications.rwpo_loss_fn,
      model,
      dim,
      2.0,
      10.0,
      0.01,
      0.01,
      time_batch_size,
      "quadratic",
      1.0,
    )
  if problem == "fp":
    return partial(fp_ou_loss, model, dim, time_batch_size)
  raise ValueError(f"Unknown problem: {problem}")


def sample_peak_rss(call):
  stop = threading.Event()
  peak = [rss_bytes()]

  def monitor():
    while not stop.is_set():
      peak[0] = max(peak[0], rss_bytes())
      time.sleep(0.001)

  thread = threading.Thread(target=monitor, daemon=True)
  thread.start()
  try:
    result = synchronize(call())
    peak[0] = max(peak[0], rss_bytes())
  finally:
    stop.set()
    thread.join()
  return result, peak[0]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--problem", choices=("ot", "rwpo", "fp"), required=True)
  parser.add_argument("--dimension", type=int, required=True)
  parser.add_argument("--batch-size", type=int, default=2048)
  parser.add_argument("--time-batch-size", type=int, default=1)
  parser.add_argument("--lambda", dest="penalty", type=float, default=5000.0)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()

  model = RQSFlow(
    event_shape=(args.dimension,),
    num_layers=2,
    hidden_sizes=[16, 16],
    num_bins=5,
    periodized=False,
  )
  model = hk.without_apply_rng(hk.multi_transform(model))

  key = jax.random.PRNGKey(args.seed)
  init_key, warmup_key, measured_key = jax.random.split(key, 3)
  params = model.init(
    init_key,
    jnp.zeros((1, args.dimension), dtype=jnp.float64),
    jnp.zeros((1,), dtype=jnp.float64),
  )
  parameter_count = sum(x.size for x in jax.tree.leaves(params))
  optimizer = optax.adam(1e-3)
  opt_state = optimizer.init(params)
  loss_fn = make_loss(args.problem, model, args.dimension, args.time_batch_size)

  @jax.jit
  def update(params_, rng_, opt_state_):
    loss, grads = jax.value_and_grad(loss_fn)(
      params_, rng_, args.penalty, args.batch_size
    )
    updates, new_opt_state = optimizer.update(grads, opt_state_)
    new_params = optax.apply_updates(params_, updates)
    return loss, new_params, new_opt_state

  rss_after_init = rss_bytes()
  compiled_update = update.lower(params, warmup_key, opt_state).compile()
  compiled_stats = compiled_update.memory_analysis()
  compiled_memory = None
  if compiled_stats is not None:
    stat_names = (
      "generated_code_size_in_bytes",
      "argument_size_in_bytes",
      "output_size_in_bytes",
      "alias_size_in_bytes",
      "temp_size_in_bytes",
      "host_argument_size_in_bytes",
      "host_output_size_in_bytes",
    )
    compiled_memory = {
      name: int(getattr(compiled_stats, name))
      for name in stat_names
      if hasattr(compiled_stats, name)
    }

  warmup_result = synchronize(compiled_update(params, warmup_key, opt_state))
  warmup_loss = float(warmup_result[0])
  del warmup_result
  gc.collect()
  rss_after_warmup = rss_bytes()

  measured_result, peak_step_rss = sample_peak_rss(
    lambda: compiled_update(params, measured_key, opt_state)
  )
  measured_loss = float(measured_result[0])
  del measured_result
  gc.collect()

  process_max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  results = {
    "problem": args.problem,
    "dimension": args.dimension,
    "parameters": parameter_count,
    "batch_size": args.batch_size,
    "interior_batch_size": args.batch_size // 32,
    "time_batch_size": args.time_batch_size,
    "preallocation": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
    "platform": platform.platform(),
    "jax": jax.__version__,
    "devices": [str(device) for device in jax.devices()],
    "warmup_loss": warmup_loss,
    "measured_loss": measured_loss,
    "memory_bytes": {
      "rss_after_init": rss_after_init,
      "rss_after_warmup": rss_after_warmup,
      "peak_rss_during_measured_step": peak_step_rss,
      "increment_over_warmup_rss": max(0, peak_step_rss - rss_after_warmup),
      "process_max_rss_including_compilation": process_max_rss,
    },
    "xla_executable_memory_bytes": compiled_memory,
  }

  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(results, indent=2) + "\n")
  print(json.dumps(results, indent=2))


if __name__ == "__main__":
  main()
