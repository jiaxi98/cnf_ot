from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.colors as mcolors
import numpy as np
import haiku as hk
from matplotlib import cm
from matplotlib import pyplot as plt

from src.types import Batch, OptState, PRNGKey


def calc_kinetic_energy(
  sample_fn,
  forward_fn,
  inverse_fn,
  params: hk.Params,
  rng: PRNGKey,
  batch_size: int = 4096,
  t_size: int = 1000,
  dim: int = 1
):

  t_array = jnp.linspace(0, 1, t_size)
  kinetic_energy = 0
  for t in t_array:

    key, rng = jax.random.split(rng)
    fake_cond_ = np.ones((batch_size, 1)) * t
    samples = sample_fn(
      params, seed=key, sample_shape=(batch_size, ), cond=fake_cond_
    )
    xi = inverse_fn(params, samples, fake_cond_)
    velocity = jax.jacfwd(partial(forward_fn, params, xi))(fake_cond_)
    kinetic_energy += jnp.mean(
      velocity[jnp.arange(batch_size), :,
               jnp.arange(batch_size), 0]**2
    )

  return kinetic_energy / t_size * dim


def plot_distribution_trajectory(
  sample_fn: callable,
  forward_fn: callable,
  params: hk.Params,
  key: PRNGKey,
  batch_size,
  mu1: float,
  mu2: float,
  var1: float,
  var2: float,
  fig_name: str = "dist_traj"
):
  t_array = jnp.linspace(0.05, 0.95, 6)
  cmap = plt.cm.Reds
  norm = mcolors.Normalize(vmin=-.5, vmax=1.5)

  plt.clf()
  plt.subplot(131)
  for i in range(6):
    fake_cond = np.ones((batch_size, 1)) * t_array[i]
    samples = sample_fn(
      params, seed=key, sample_shape=(batch_size, ), cond=fake_cond
    )
    plt.scatter(
      samples[..., 0], samples[..., 1], s=.1, color=cmap(norm(t_array[i]))
    )
  plt.subplot(132)
  x = jnp.linspace(-3, 3, 100)
  y = jnp.linspace(-3, 3, 100)
  xy = jnp.array(jnp.meshgrid(x, y))
  xy = jnp.transpose(jnp.reshape(xy, (2, 10000)))
  fake_cond = jnp.zeros_like(xy[:, 0:1])
  xy_forward = forward_fn(params, xy, jnp.zeros(1))
  xy_correct = mu1 + xy * jnp.sqrt(var1)
  err = jnp.sum((xy_forward - xy_correct)**2, axis=1)
  plt.imshow(jnp.reshape(err, (100, 100)))
  plt.axis("off")
  plt.colorbar(orientation="horizontal", fraction=0.2)

  plt.subplot(133)
  fake_cond = jnp.ones_like(xy[:, 0:1])
  xy_forward = forward_fn(params, xy, jnp.zeros(1))
  xy_correct = mu2 + xy * jnp.sqrt(var2)
  err = jnp.sum((xy_forward - xy_correct)**2, axis=1)
  plt.imshow(jnp.reshape(err, (100, 100)))
  plt.axis("off")
  plt.colorbar(orientation="horizontal", fraction=0.2)
  plt.suptitle(
    r"$\rho_0 \sim N(({},{})^T,I), \rho_1 \sim N(({},{})^T,I)$".format(
      mu1[0], mu1[1], mu2[0], mu2[1]
    )
  )
  plt.savefig("results/fig/" + fig_name + ".pdf")
  plt.clf()


def plot_samples_snapshot(
  sample_fn: callable,
  params: hk.Params,
  key: PRNGKey,
  batch_size,
  t_array: jnp.array,
):

  plt.clf()
  plt.figure(figsize=(10, 2))
  cmap = plt.cm.Reds
  norm = mcolors.Normalize(vmin=-.5, vmax=1.5)

  for i in range(5):
    plt.subplot(1, 5, i + 1)
    fake_cond = np.ones((batch_size, 1)) * t_array[i]
    samples = sample_fn(
      params, seed=key, sample_shape=(batch_size, ), cond=fake_cond
    )
    plt.scatter(
      samples[..., 0],
      samples[..., 1],
      s=1,
      label=str(t_array[i]),
      color=cmap(norm(t_array[i]))
    )
    plt.legend()

  plt.savefig("results/fig/samples.pdf")
  plt.clf()


def plot_density_snapshot(
  log_prob_fn: callable,
  params: hk.Params,
  t_array = jnp.linspace(0, 1, 5),
):

  plt.clf()
  plt.figure(figsize=(10, 2))
  cmap = plt.cm.Reds
  norm = mcolors.Normalize(vmin=-.5, vmax=1.5)

  for i in range(5):
    plt.subplot(1, 5, i + 1)
    x_min = -8
    x_max = 8
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(x_min, x_max, 100)
    X, Y = np.meshgrid(x, y)
    XY = jnp.hstack([X.reshape(100**2, 1), Y.reshape(100**2, 1)])
    fake_cond_ = np.ones((1, )) * t_array[i]
    log_prob = log_prob_fn(params, XY, cond=fake_cond_)
    plt.imshow(jnp.exp(log_prob.reshape(100, 100)))
    plt.title("t = {:.2f}".format(t_array[i]))

  plt.savefig("results/fig/density.pdf")
  plt.clf()


def plot_trajectory(
  forward_fn: callable,
  inverse_fn: callable,
  log_prob_fn: callable,
  params: hk.Params,
  r_: jnp.array,
  t_array: jnp.array,
):
  
  plt.clf()
  x_min = -8
  x_max = 8
  x = np.linspace(x_min, x_max, 100)
  y = np.linspace(x_min, x_max, 100)
  X, Y = np.meshgrid(x, y)
  XY = jnp.hstack([X.reshape(100**2, 1), Y.reshape(100**2, 1)])
  fake_cond_ = np.zeros((1, ))
  log_prob = log_prob_fn(params, XY, cond=fake_cond_)
  plt.imshow(jnp.exp(log_prob.reshape(100, 100)), cmap=cm.viridis)
  
  xi = inverse_fn(params, r_, jnp.zeros(1))
  colors = ["red", "green", "blue", "yellow"]
  for t in t_array:
    r_ = forward_fn(params, xi, jnp.ones(1) * t)
    plt.scatter((r_[:, 0]+8)/16*100, (r_[:, 1]+8)/16*100, c="red", s=5)
  plt.savefig("results/fig/traj.pdf")


def plot_traj_and_velocity(
  sample_fn,
  forward_fn,
  inverse_fn,
  params: hk.Params,
  rng: PRNGKey,
  t_array: jnp.array,
  quiver_size: float = .1
):

  batch_size_pdf = 1024
  batch_size_velocity = 64
  fig1 = plt.figure(figsize=(10, 10))
  fig2 = plt.figure(figsize=(10, 10))
  ax1 = fig1.subplots(3, 2)
  ax2 = fig2.subplots(3, 2)
  i = 0
  for t in t_array:

    key, rng = jax.random.split(rng)
    fake_cond_ = np.ones((batch_size_pdf, 1)) * t
    samples = sample_fn(
      params, seed=key, sample_shape=(batch_size_pdf, ), cond=fake_cond_
    )
    ax1[i // 2, i % 2].scatter(samples[..., 0], samples[..., 1], s=1)

    fake_cond_ = np.ones((batch_size_velocity, 1)) * t
    x_min = -8
    x_max = 8
    x = np.linspace(x_min, x_max, 10)
    y = np.linspace(x_min, x_max, 10)
    X, Y = np.meshgrid(x, y)
    XY = jnp.hstack([X.reshape(10**2, 1), Y.reshape(10**2, 1)])
    # samples = sample_fn(params, seed=key, sample_shape=(batch_size_velocity, ), cond=fake_cond_)
    xi = inverse_fn(params, XY, jnp.zeros(1))
    velocity = jax.jacfwd(partial(forward_fn, params, xi))(jnp.zeros(1))
    ax2[i // 2, i % 2].quiver(
      XY[..., 0],
      XY[..., 1],
      velocity[:, 0, 0],
      velocity[:, 1, 0],
    )
    #scale=quiver_size)
    i += 1
  plt.savefig("results/fig/traj.pdf")
  plt.clf()


def plot_1d_map(
  forward_fn,
  params: hk.Params,
  final_mean,
):
  t_array = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
  batch_size_pdf = 1024
  fig1 = plt.figure(figsize=(10, 10))
  ax1 = fig1.subplots(3, 2)
  i = 0
  for t in t_array:
    fake_cond_ = np.ones((batch_size_pdf, 1)) * t
    x_axis = np.linspace(-3, 3, batch_size_pdf).reshape(-1, 1)
    y_axis = forward_fn(params, x_axis, fake_cond_)
    true_y = x_axis + final_mean * t
    ax1[i // 2, i % 2].plot(x_axis, y_axis, "b")
    ax1[i // 2, i % 2].plot(x_axis, true_y, "r")
    i += 1
  plt.savefig("results/fig/mapping_1d.pdf")
  # plt.clf()
  plt.show()
