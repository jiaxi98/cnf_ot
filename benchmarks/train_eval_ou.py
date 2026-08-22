"""Train and evaluate the two-dimensional OU experiment from the paper."""

import argparse
import json
import math
import pickle
import time
from functools import partial
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from cnf_ot.models.flows import RQSFlow


jax.config.update("jax_enable_x64", True)


def gaussian_log_prob(x, variance):
  dim = x.shape[-1]
  return -0.5 * (
    dim * jnp.log(2 * jnp.pi * variance) + jnp.sum(x**2, axis=-1) / variance
  )


def flow_matching_loss(model, params, cond, rng, batch_size, a, gamma, dt, dx):
  cond_batch = lambda value: jnp.full((batch_size, 1), value)
  before = model.apply.sample(
    params,
    seed=rng,
    sample_shape=(batch_size,),
    cond=cond_batch(cond - dt / 2),
  )
  after = model.apply.sample(
    params,
    seed=rng,
    sample_shape=(batch_size,),
    cond=cond_batch(cond + dt / 2),
  )
  samples = model.apply.sample(
    params,
    seed=rng,
    sample_shape=(batch_size,),
    cond=cond_batch(cond),
  )

  current_velocity = (after - before) / dt
  score = jnp.zeros_like(samples)
  for coordinate in range(samples.shape[-1]):
    offset = jnp.zeros((1, samples.shape[-1]))
    offset = offset.at[0, coordinate].set(dx / 2)
    log_prob_plus = model.apply.log_prob(
      params, samples + offset, cond=jnp.full((1,), cond)
    )
    log_prob_minus = model.apply.log_prob(
      params, samples - offset, cond=jnp.full((1,), cond)
    )
    score = score.at[:, coordinate].set(
      (log_prob_plus - log_prob_minus) / dx
    )

  fp_drift = current_velocity + gamma * score
  true_drift = -a * samples
  return jnp.mean(jnp.sum((fp_drift - true_drift) ** 2, axis=-1)) / 2


def make_loss(
  model,
  dim,
  time_batch_size,
  boundary_batch_size,
  interior_batch_size,
  penalty,
  initial_variance,
  horizon,
  a,
  gamma,
  dt,
  dx,
):
  def loss_fn(params, rng):
    keys = jax.random.split(rng, time_batch_size + 2)
    boundary_key, time_key = keys[:2]
    interior_keys = keys[2:]

    cond_zero = jnp.zeros((boundary_batch_size, 1))
    boundary_samples, boundary_log_prob = model.apply.sample_and_log_prob(
      params,
      cond=cond_zero,
      seed=boundary_key,
      sample_shape=(boundary_batch_size,),
    )
    boundary_loss = jnp.mean(
      boundary_log_prob
      - gaussian_log_prob(boundary_samples, initial_variance)
    )

    times = jax.random.uniform(time_key, (time_batch_size,)) * horizon
    interior_loss = 0.0
    for cond, interior_key in zip(times, interior_keys):
      interior_loss += flow_matching_loss(
        model,
        params,
        cond,
        interior_key,
        interior_batch_size,
        a,
        gamma,
        dt,
        dx,
      )

    return penalty * boundary_loss + horizon * interior_loss / time_batch_size

  return loss_fn


def exact_variance(t, initial_variance, a, gamma):
  stationary_variance = gamma / a
  return stationary_variance + (
    initial_variance - stationary_variance
  ) * jnp.exp(-2 * a * t)


def make_evaluator(model, dim, initial_variance, a, gamma, dt, dx):
  @partial(jax.jit, static_argnames=("batch_size",))
  def evaluate_batch(params, rng, cond, batch_size):
    variance = exact_variance(cond, initial_variance, a, gamma)
    samples = jax.random.normal(rng, (batch_size, dim)) * jnp.sqrt(variance)
    cond_value = jnp.full((1,), cond)

    learned_log_prob = model.apply.log_prob(params, samples, cond=cond_value)
    true_log_prob = gaussian_log_prob(samples, variance)
    true_density = jnp.exp(true_log_prob)
    density_ratio_error = jnp.expm1(learned_log_prob - true_log_prob)
    density_error_numerator = jnp.sum(
      true_density * density_ratio_error**2
    )
    density_error_denominator = jnp.sum(true_density)

    learned_score = jnp.zeros_like(samples)
    for coordinate in range(dim):
      offset = jnp.zeros((1, dim)).at[0, coordinate].set(dx / 2)
      log_prob_plus = model.apply.log_prob(
        params, samples + offset, cond=cond_value
      )
      log_prob_minus = model.apply.log_prob(
        params, samples - offset, cond=cond_value
      )
      learned_score = learned_score.at[:, coordinate].set(
        (log_prob_plus - log_prob_minus) / dx
      )
    true_score = -samples / variance

    latent = model.apply.inverse(params, samples, cond_value)
    before = model.apply.forward(
      params, latent, jnp.full((1,), cond - dt / 2)
    )
    after = model.apply.forward(
      params, latent, jnp.full((1,), cond + dt / 2)
    )
    learned_current_velocity = (after - before) / dt
    true_current_velocity = (-a + gamma / variance) * samples

    learned_drift = learned_current_velocity + gamma * learned_score
    true_drift = -a * samples

    def vector_error_sums(learned, exact):
      numerator = jnp.sum((learned - exact) ** 2)
      denominator = jnp.sum(exact**2)
      return numerator, denominator

    return (
      density_error_numerator,
      density_error_denominator,
      *vector_error_sums(learned_score, true_score),
      *vector_error_sums(learned_current_velocity, true_current_velocity),
      *vector_error_sums(learned_drift, true_drift),
    )

  return evaluate_batch


def relative_errors(statistics):
  values = [float(value) for value in statistics]
  return {
    "relative_l2_density_error": math.sqrt(values[0] / values[1]),
    "relative_score_rmse": math.sqrt(values[2] / values[3]),
    "relative_current_velocity_rmse": math.sqrt(values[4] / values[5]),
    "relative_fp_drift_rmse": math.sqrt(values[6] / values[7]),
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--epochs", type=int, default=30_000)
  parser.add_argument("--time-batch-size", type=int, default=20)
  parser.add_argument("--boundary-batch-size", type=int, default=2048)
  parser.add_argument("--interior-batch-size", type=int, default=64)
  parser.add_argument("--eval-batch-size", type=int, default=131_072)
  parser.add_argument("--chunk-size", type=int, default=100)
  parser.add_argument("--lambda", dest="penalty", type=float, default=5000.0)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  args = parser.parse_args()

  dim = 2
  horizon = 1.0
  initial_variance = 4.0
  a = 1.0
  gamma = 0.5
  dt = 0.01
  dx = 0.01

  if args.epochs % args.chunk_size:
    raise ValueError("epochs must be divisible by chunk-size")

  model = RQSFlow(
    event_shape=(dim,),
    num_layers=2,
    hidden_sizes=[16, 16],
    num_bins=5,
    periodized=False,
  )
  model = hk.without_apply_rng(hk.multi_transform(model))
  rng = jax.random.PRNGKey(args.seed)
  init_key, train_key, eval_key = jax.random.split(rng, 3)
  params = model.init(
    init_key,
    jnp.zeros((1, dim), dtype=jnp.float64),
    jnp.zeros((1,), dtype=jnp.float64),
  )
  parameter_count = sum(value.size for value in jax.tree.leaves(params))

  optimizer = optax.adam(1e-3)
  opt_state = optimizer.init(params)
  loss_fn = make_loss(
    model,
    dim,
    args.time_batch_size,
    args.boundary_batch_size,
    args.interior_batch_size,
    args.penalty,
    initial_variance,
    horizon,
    a,
    gamma,
    dt,
    dx,
  )

  @partial(jax.jit, static_argnames=("steps",))
  def train_chunk(params_, opt_state_, rng_, steps):
    def update(carry, _):
      params_value, opt_state_value, key = carry
      key, step_key = jax.random.split(key)
      loss, grads = jax.value_and_grad(loss_fn)(params_value, step_key)
      updates, opt_state_value = optimizer.update(
        grads, opt_state_value, params_value
      )
      params_value = optax.apply_updates(params_value, updates)
      return (params_value, opt_state_value, key), loss

    return jax.lax.scan(
      update, (params_, opt_state_, rng_), xs=None, length=steps
    )

  started_at = time.perf_counter()
  final_loss = math.nan
  num_chunks = args.epochs // args.chunk_size
  for chunk_index in range(num_chunks):
    (params, opt_state, train_key), losses = train_chunk(
      params, opt_state, train_key, steps=args.chunk_size
    )
    final_loss = float(losses[-1].block_until_ready())
    completed_steps = (chunk_index + 1) * args.chunk_size
    if completed_steps % 1000 == 0 or completed_steps == args.epochs:
      print(f"step={completed_steps} loss={final_loss:.8e}", flush=True)
  training_seconds = time.perf_counter() - started_at

  evaluate_batch = make_evaluator(
    model, dim, initial_variance, a, gamma, dt, dx
  )
  evaluation = []
  for cond in (0.25, 0.5, 0.75, 1.0):
    eval_key, time_key = jax.random.split(eval_key)
    statistics = evaluate_batch(
      params, time_key, cond, args.eval_batch_size
    )
    statistics = jax.tree.map(lambda value: value.block_until_ready(), statistics)
    row = {"time": cond, **relative_errors(statistics)}
    evaluation.append(row)
    print(json.dumps(row), flush=True)

  results = {
    "problem": "two-dimensional Ornstein-Uhlenbeck Fokker-Planck equation",
    "architecture": {
      "autoregressive_layers": 2,
      "conditioner_hidden_sizes": [16, 16],
      "spline_bins": 5,
      "parameters": parameter_count,
    },
    "training": {
      "epochs": args.epochs,
      "learning_rate": 0.001,
      "penalty": args.penalty,
      "time_batch_size": args.time_batch_size,
      "boundary_batch_size": args.boundary_batch_size,
      "interior_batch_size": args.interior_batch_size,
      "seed": args.seed,
      "final_stochastic_loss": final_loss,
      "seconds": training_seconds,
    },
    "equation": {
      "horizon": horizon,
      "initial_variance": initial_variance,
      "drift_coefficient": a,
      "diffusion_coefficient": gamma,
    },
    "evaluation": {
      "batch_size_per_time": args.eval_batch_size,
      "space_finite_difference_step": dx,
      "time_finite_difference_step": dt,
      "sampling_distribution": "exact OU density at each reported time",
      "results": evaluation,
    },
    "jax": jax.__version__,
    "devices": [str(device) for device in jax.devices()],
  }

  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(results, indent=2) + "\n")
  args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
  with args.checkpoint.open("wb") as checkpoint_file:
    pickle.dump(jax.device_get(params), checkpoint_file)
  print(json.dumps(results, indent=2))


if __name__ == "__main__":
  main()
