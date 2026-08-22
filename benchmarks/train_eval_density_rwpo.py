"""Train and evaluate a two-dimensional density-coupled RWPO problem."""

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


def exact_variance(t, horizon, beta):
  return 2 * (horizon - t + 1) / beta


def exact_shift_constant(dim, horizon, beta, kappa):
  if dim != 2:
    raise ValueError("The closed-form shift is implemented only for d=2")
  return kappa * beta * math.log(horizon + 1) / (16 * math.pi)


def score_finite_difference(model, params, samples, cond, dim, dx):
  score = jnp.zeros_like(samples)
  cond_value = jnp.full((1,), cond)
  for coordinate in range(dim):
    offset = jnp.zeros((1, dim)).at[0, coordinate].set(dx / 2)
    log_prob_plus = model.apply.log_prob(
      params, samples + offset, cond=cond_value
    )
    log_prob_minus = model.apply.log_prob(
      params, samples - offset, cond=cond_value
    )
    score = score.at[:, coordinate].set(
      (log_prob_plus - log_prob_minus) / dx
    )
  return score


def running_loss(
  model,
  params,
  cond,
  rng,
  batch_size,
  dim,
  horizon,
  beta,
  kappa,
  dt,
  dx,
):
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
  samples, learned_log_prob = model.apply.sample_and_log_prob(
    params,
    seed=rng,
    sample_shape=(batch_size,),
    cond=cond_batch(cond),
  )

  current_velocity = (after - before) / dt
  score = score_finite_difference(
    model, params, samples, cond, dim, dx
  )
  fp_drift = current_velocity + score / beta
  kinetic = jnp.mean(jnp.sum(fp_drift**2, axis=-1)) / 2

  true_log_prob = gaussian_log_prob(
    samples, exact_variance(cond, horizon, beta)
  )
  density_coupling = kappa * jnp.mean(
    0.5 * jnp.exp(learned_log_prob) - jnp.exp(true_log_prob)
  )
  return kinetic + density_coupling


def make_loss(
  model,
  dim,
  time_batch_size,
  boundary_batch_size,
  interior_batch_size,
  penalty,
  horizon,
  beta,
  kappa,
  dt,
  dx,
):
  initial_variance = exact_variance(0.0, horizon, beta)

  def loss_fn(params, rng):
    keys = jax.random.split(rng, time_batch_size + 3)
    boundary_key, terminal_key, time_key = keys[:3]
    interior_keys = keys[3:]

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

    cond_terminal = jnp.full((boundary_batch_size, 1), horizon)
    terminal_samples = model.apply.sample(
      params,
      cond=cond_terminal,
      seed=terminal_key,
      sample_shape=(boundary_batch_size,),
    )
    terminal_loss = jnp.mean(
      jnp.sum(terminal_samples**2, axis=-1)
    ) / 2

    times = jax.random.uniform(time_key, (time_batch_size,)) * horizon
    interior_loss = 0.0
    for cond, interior_key in zip(times, interior_keys):
      interior_loss += running_loss(
        model,
        params,
        cond,
        interior_key,
        interior_batch_size,
        dim,
        horizon,
        beta,
        kappa,
        dt,
        dx,
      )

    return (
      penalty * boundary_loss
      + horizon * interior_loss / time_batch_size
      + terminal_loss
    )

  return loss_fn


def make_component_evaluator(model, dim, horizon, beta, dt, dx):
  @partial(jax.jit, static_argnames=("batch_size",))
  def evaluate_batch(params, rng, cond, batch_size):
    variance = exact_variance(cond, horizon, beta)
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

    learned_score = score_finite_difference(
      model, params, samples, cond, dim, dx
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
    time_to_go = horizon - cond + 1
    true_current_velocity = -samples / (2 * time_to_go)

    learned_drift = learned_current_velocity + learned_score / beta
    true_drift = -samples / time_to_go

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


def make_objective_evaluators(
  model, dim, horizon, beta, kappa, dt, dx
):
  @partial(jax.jit, static_argnames=("batch_size",))
  def running_components(params, rng, cond, batch_size):
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
    samples, learned_log_prob = model.apply.sample_and_log_prob(
      params,
      seed=rng,
      sample_shape=(batch_size,),
      cond=cond_batch(cond),
    )
    current_velocity = (after - before) / dt
    score = score_finite_difference(
      model, params, samples, cond, dim, dx
    )
    drift = current_velocity + score / beta
    kinetic = jnp.mean(jnp.sum(drift**2, axis=-1)) / 2
    true_log_prob = gaussian_log_prob(
      samples, exact_variance(cond, horizon, beta)
    )
    coupling = kappa * jnp.mean(
      0.5 * jnp.exp(learned_log_prob) - jnp.exp(true_log_prob)
    )
    return kinetic, coupling

  @partial(jax.jit, static_argnames=("batch_size",))
  def endpoint_components(params, rng, batch_size):
    boundary_key, terminal_key = jax.random.split(rng)
    initial_variance = exact_variance(0.0, horizon, beta)
    boundary_samples, boundary_log_prob = model.apply.sample_and_log_prob(
      params,
      cond=jnp.zeros((batch_size, 1)),
      seed=boundary_key,
      sample_shape=(batch_size,),
    )
    boundary_kl = jnp.mean(
      boundary_log_prob
      - gaussian_log_prob(boundary_samples, initial_variance)
    )
    terminal_samples = model.apply.sample(
      params,
      cond=jnp.full((batch_size, 1), horizon),
      seed=terminal_key,
      sample_shape=(batch_size,),
    )
    terminal_potential = jnp.mean(
      jnp.sum(terminal_samples**2, axis=-1)
    ) / 2
    return boundary_kl, terminal_potential

  return running_components, endpoint_components


def evaluate_objective(
  model,
  params,
  rng,
  dim,
  horizon,
  beta,
  kappa,
  dt,
  dx,
  time_points,
  batch_size,
):
  running_components, endpoint_components = make_objective_evaluators(
    model, dim, horizon, beta, kappa, dt, dx
  )
  keys = jax.random.split(rng, time_points + 1)
  kinetic_sum = 0.0
  coupling_sum = 0.0
  for index in range(time_points):
    cond = horizon * (index + 0.5) / time_points
    kinetic, coupling = running_components(
      params, keys[index], cond, batch_size
    )
    kinetic_sum += float(kinetic.block_until_ready())
    coupling_sum += float(coupling.block_until_ready())

  boundary_kl, terminal_potential = endpoint_components(
    params, keys[-1], batch_size
  )
  boundary_kl = float(boundary_kl.block_until_ready())
  terminal_potential = float(terminal_potential.block_until_ready())
  kinetic = horizon * kinetic_sum / time_points
  density_coupling = horizon * coupling_sum / time_points
  raw_objective = kinetic + density_coupling + terminal_potential
  shift = exact_shift_constant(dim, horizon, beta, kappa)
  shifted_objective = raw_objective + shift
  exact_shifted_objective = dim * (math.log(horizon + 1) + 1) / beta
  exact_raw_objective = exact_shifted_objective - shift
  return {
    "kinetic": kinetic,
    "density_coupling": density_coupling,
    "terminal_potential": terminal_potential,
    "boundary_kl": boundary_kl,
    "raw_objective": raw_objective,
    "exact_raw_objective": exact_raw_objective,
    "raw_objective_relative_error": abs(
      raw_objective - exact_raw_objective
    ) / abs(exact_raw_objective),
    "additive_shift": shift,
    "shifted_objective": shifted_objective,
    "exact_shifted_objective": exact_shifted_objective,
    "shifted_objective_relative_error": abs(
      shifted_objective - exact_shifted_objective
    ) / exact_shifted_objective,
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--epochs", type=int, default=30_000)
  parser.add_argument("--time-batch-size", type=int, default=20)
  parser.add_argument("--boundary-batch-size", type=int, default=2048)
  parser.add_argument("--interior-batch-size", type=int, default=64)
  parser.add_argument("--eval-batch-size", type=int, default=131_072)
  parser.add_argument("--objective-batch-size", type=int, default=8192)
  parser.add_argument("--objective-time-points", type=int, default=100)
  parser.add_argument("--chunk-size", type=int, default=100)
  parser.add_argument("--lambda", dest="penalty", type=float, default=200.0)
  parser.add_argument("--kappa", type=float, default=10.0)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  args = parser.parse_args()

  dim = 2
  horizon = 2.0
  beta = 1.0
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
  init_key, train_key, component_key, objective_key = jax.random.split(rng, 4)
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
    horizon,
    beta,
    args.kappa,
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

  evaluate_batch = make_component_evaluator(
    model, dim, horizon, beta, dt, dx
  )
  component_evaluation = []
  for cond in (0.5, 1.0, 1.5, 2.0):
    component_key, time_key = jax.random.split(component_key)
    statistics = evaluate_batch(
      params, time_key, cond, args.eval_batch_size
    )
    statistics = jax.tree.map(
      lambda value: value.block_until_ready(), statistics
    )
    row = {"time": cond, **relative_errors(statistics)}
    component_evaluation.append(row)
    print(json.dumps(row), flush=True)

  objective = evaluate_objective(
    model,
    params,
    objective_key,
    dim,
    horizon,
    beta,
    args.kappa,
    dt,
    dx,
    args.objective_time_points,
    args.objective_batch_size,
  )

  results = {
    "problem": "two-dimensional density-coupled quadratic RWPO",
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
      "density_coupling": args.kappa,
      "time_batch_size": args.time_batch_size,
      "boundary_batch_size": args.boundary_batch_size,
      "interior_batch_size": args.interior_batch_size,
      "seed": args.seed,
      "final_stochastic_loss": final_loss,
      "seconds": training_seconds,
    },
    "equation": {
      "horizon": horizon,
      "inverse_diffusion_coefficient": beta,
      "initial_variance": exact_variance(0.0, horizon, beta),
      "terminal_potential": "0.5 * ||x||^2",
      "running_density_term": "kappa * (0.5 * p - p_star)",
    },
    "evaluation": {
      "batch_size_per_time": args.eval_batch_size,
      "objective_batch_size": args.objective_batch_size,
      "objective_time_points": args.objective_time_points,
      "space_finite_difference_step": dx,
      "time_finite_difference_step": dt,
      "sampling_distribution_for_component_errors": (
        "exact RWPO density at each reported time"
      ),
      "component_errors": component_evaluation,
      "objective": objective,
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
