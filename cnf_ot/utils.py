from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.colors as mcolors
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

from cnf_ot.types import PRNGKey


def calc_kinetic_energy(
  sample_fn,
  params: hk.Params,
  rng: PRNGKey,
  batch_size: int = 65536,
  t_size: int = 10000,
  dim: int = 1
):
  """Calculate kinetic energy via Monte Carlo sampling
  """

  t_array = jnp.linspace(0, 1, t_size)
  e_kin = 0
  dt = 0.01

  for t in t_array:

    _rng, rng = jax.random.split(rng)
    fake_cond_ = np.ones((batch_size, 1)) * (t - dt / 2)
    r1 = sample_fn(
      params, seed=_rng, sample_shape=(batch_size, ), cond=fake_cond_
    )
    fake_cond_ = np.ones((batch_size, 1)) * (t + dt / 2)
    r2 = sample_fn(
      params, seed=_rng, sample_shape=(batch_size, ), cond=fake_cond_
    )
    velocity = (r2 - r1) / dt
    e_kin += jnp.mean(velocity**2) / 2

  return e_kin / t_size * dim


def calc_score_kinetic_energy(
  sample_fn,
  log_prob_fn,
  params: hk.Params,
  T: float = 1,
  beta: float = 1,
  dim: int = 1,
  rng: PRNGKey = PRNGKey(0),
  batch_size: int = 65536,
  t_size: int = 10000,
):
  """Calculate kinetic energy with velocity corrected by score function
  """

  t_array = jnp.linspace(0, T, t_size)
  e_kin = 0
  dt = 0.01

  for t in t_array:

    _rng, rng = jax.random.split(rng)
    fake_cond_ = np.ones((batch_size, 1)) * (t - dt / 2)
    r1 = sample_fn(
      params, seed=_rng, sample_shape=(batch_size, ), cond=fake_cond_
    )
    fake_cond_ = np.ones((batch_size, 1)) * (t + dt / 2)
    r2 = sample_fn(
      params, seed=_rng, sample_shape=(batch_size, ), cond=fake_cond_
    )
    fake_cond_ = np.ones((batch_size, 1)) * t
    r3 = sample_fn(
      params, seed=_rng, sample_shape=(batch_size, ), cond=fake_cond_
    )
    velocity = (r2 - r1) / dt
    score = jnp.zeros((batch_size, dim))
    dx = 0.01
    for i in range(dim):
      dr = jnp.zeros((1, dim))
      dr = dr.at[0, i].set(dx / 2)
      log_p1 = log_prob_fn(params, r3 + dr, cond=jnp.ones(1) * t)
      log_p2 = log_prob_fn(params, r3 - dr, cond=jnp.ones(1) * t)
      score = score.at[:, i].set((log_p1 - log_p2) / dx)
    velocity += score / beta
    plt.quiver
    e_kin += jnp.mean(velocity**2) / 2

  return e_kin / t_size * dim


def plot_velocity_field(
  log_prob_fn: callable,
  params: hk.Params,
  r_: jnp.array,
  _score: str = False,
):
  """Visualize the velocity field for debug
  """

  plt.clf()
  fig, ax = plt.subplots(1, 1, figsize=(5, 5))
  # axs = axs.flatten()
  x_min = -5
  x_max = 5
  x = np.linspace(x_min, x_max, 100)
  y = np.linspace(x_min, x_max, 100)
  X, Y = np.meshgrid(x, y)
  XY = jnp.hstack([X.reshape(100**2, 1), Y.reshape(100**2, 1)])
  dim = r_.shape[-1]

  if _score:
    fake_cond_ = np.zeros((1, ))
    log_prob = log_prob_fn(params, XY, cond=fake_cond_)
    ax.imshow(jnp.exp(log_prob.reshape(100, 100)), cmap=cm.viridis)
    field = jnp.zeros((r_.shape[0], dim))
    dx = 0.01
    for i in range(dim):
      dr = jnp.zeros((1, dim))
      dr = dr.at[0, i].set(dx / 2)
      log_p1 = log_prob_fn(params, r_ + dr, cond=jnp.ones(1) * 0)
      log_p2 = log_prob_fn(params, r_ - dr, cond=jnp.ones(1) * 0)
      field = field.at[:, i].set((log_p1 - log_p2) / dx)

  else:
    # debug for the ``smiling'' distribution
    # NOTE: ploting the velocity fields for debugging the experiments is
    # really powerful:
    # * The numerical value 1e-20 is really crucial for
    # the log-sum-exp formula for the distribution.
    # * The range of the plotting region is also important as the potential
    # function is fourth-order which blows up quickly.
    r1 = 4
    r2 = 4
    x_min = -2.5
    x_max = 2.5
    y_min = 1
    y_max = 2
    nx = 10
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, nx)
    X, Y = np.meshgrid(x, y)
    r_ = jnp.hstack([X.reshape(nx**2, 1), Y.reshape(nx**2, 1)])
    x = r_[:, 0]
    y = r_[:, 1]
    y0 = 8/5
    x0 = 3/2
    _pi = jnp.exp(-r1/4 * ((x - x0)**2 + (y - 6/5)**2 - .5)**2 -
                  r2/2 * (y - y0)**2) +\
          jnp.exp(-r1/4 * ((x + x0)**2 + (y - 6/5)**2 - .5)**2 -
                  r2/2 * (y - y0)**2) + 1e-20
    dpidx = jnp.exp(-r1/4 * ((x - x0)**2 + (y - 6/5)**2 - .5)**2 -
                  r2/2 * (y - y0)**2) * ((x - x0)**2 + (y - 6/5)**2 - .5) *\
                  r1 * (x - x0) +\
            jnp.exp(-r1/4 * ((x + x0)**2 + (y - 6/5)**2 - .5)**2 -
                    r2/2 * (y - y0)**2) * ((x + x0)**2 + (y - 6/5)**2 - .5) *\
                  r1 * (x + x0)
    dpidy = jnp.exp(-r1/4 * ((x - x0)**2 + (y - 6/5)**2 - .5)**2 -
                  r2/2 * (y - y0)**2) * (((x - x0)**2 + (y - 6/5)**2 - .5) *
                  r1 * (y - 6/5) + r2 * (y - y0)) +\
            jnp.exp(-r1/4 * ((x + x0)**2 + (y - 6/5)**2 - .5)**2 -
                    r2/2 * (y - y0)**2) * (((x + x0)**2 + (y - 6/5)**2 - .5) *
                  r1 * (y - 6/5) + r2 * (y - y0))
    field = -jnp.concat([(dpidx/_pi)[:, None], (dpidy/_pi)[:, None]], axis=1)
    # grad_x = -(x**2 + y**2 - 4) * r1 * x
    # grad_y = -(x**2 + y**2 - 4) * r1 * y - r2 * (y + 1)
    # field = jnp.concat([grad_x[:, None], grad_y[:, None]], axis=1)

  ax.quiver(r_[:, 0], r_[:, 1], field[:, 0], field[:, 1])
  # ax.axis("off")
  # fig.tight_layout(pad=0.2)
  # plt.subplots_adjust(hspace=0.1)
  plt.savefig("results/fig/field.pdf")
  plt.clf()


def plot_distribution_trajectory(
  sample_fn: callable,
  forward_fn: callable,
  params: hk.Params,
  rng: PRNGKey,
  batch_size,
  mu1: float,
  mu2: float,
  var1: float,
  var2: float,
  fig_name: str = "dist_traj"
):
  """Deprecated ploting function
  TODO: can delete
  """

  t_array = jnp.linspace(0.05, 0.95, 6)
  cmap = plt.cm.Reds
  norm = mcolors.Normalize(vmin=-.5, vmax=1.5)

  plt.clf()
  plt.subplot(131)
  for i in range(6):
    fake_cond = np.ones((batch_size, 1)) * t_array[i]
    samples = sample_fn(
      params, seed=rng, sample_shape=(batch_size, ), cond=fake_cond
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
  rng: PRNGKey,
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
      params, seed=rng, sample_shape=(batch_size, ), cond=fake_cond
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
  t_array=jnp.linspace(0, 1, 10),
):

  plt.clf()
  plt.figure(figsize=(10, 2))

  for i in range(10):
    plt.subplot(2, 5, i + 1)
    x_min = -6
    x_max = 6
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(x_min, x_max, 100)
    X, Y = np.meshgrid(x, y)
    XY = jnp.hstack([X.reshape(100**2, 1), Y.reshape(100**2, 1)])
    fake_cond_ = np.ones((1, )) * t_array[i]
    log_prob = log_prob_fn(params, XY, cond=fake_cond_)
    plt.imshow(jnp.exp(log_prob.reshape(100, 100)))
    plt.axis("off")
    plt.title("t = {:.2f}".format(t_array[i]))

  plt.savefig("results/fig/density.pdf")
  plt.clf()


def plot_density_and_trajectory(
  forward_fn: callable,
  inverse_fn: callable,
  log_prob_fn: callable,
  params: hk.Params,
  r_: jnp.array,
  t_array: jnp.array,
):

  plt.clf()
  fig, axs = plt.subplots(2, 5, figsize=(5, 2))
  axs = axs.flatten()
  x_min = -10
  x_max = 10
  # x_min = -3
  # x_max = 3
  x = np.linspace(x_min, x_max, 100)
  y = np.linspace(x_min, x_max, 100)
  X, Y = np.meshgrid(x, y)
  XY = jnp.hstack([X.reshape(100**2, 1), Y.reshape(100**2, 1)])
  xi = inverse_fn(params, r_, jnp.zeros(1))

  for i in range(len(t_array)):
    fake_cond_ = t_array[i] * jnp.ones((1, ))
    log_prob = log_prob_fn(params, XY, cond=fake_cond_)
    axs[i].imshow(jnp.exp(log_prob.reshape(100, 100)), cmap=cm.viridis)
    for t in t_array:
      r_ = forward_fn(params, xi, jnp.ones(1) * t)
      axs[i].scatter(
        (r_[:, 0] + x_max) / 2 / x_max * 100,
        (r_[:, 1] + x_max) / 2 / x_max * 100,
        c="red",
        marker='.',
        s=.1
      )
    axs[i].axis("off")

  fig.tight_layout(pad=0.2)
  # plt.subplots_adjust(hspace=0.1)
  plt.savefig("results/fig/traj.pdf")


def plot_high_dim_density_and_trajectory(
  forward_fn: callable,
  inverse_fn: callable,
  log_prob_fn: callable,
  params: hk.Params,
  r_: jnp.array,
  t_array: jnp.array,
):

  plt.clf()
  fig, axs = plt.subplots(2, 5, figsize=(5, 2))
  axs = axs.flatten()
  x_min = -10
  x_max = 10
  x = np.linspace(x_min, x_max, 100)
  y = np.linspace(x_min, x_max, 100)
  X, Y = np.meshgrid(x, y)
  XY = jnp.hstack(
    [X.reshape(100**2, 1), Y.reshape(100**2, 1), jnp.zeros((100**2, 1))]
  )
  xi = inverse_fn(params, r_, jnp.zeros(1))

  for i in range(10):
    fake_cond_ = jnp.ones((1, )) * i * .1
    log_prob = log_prob_fn(params, XY, cond=fake_cond_)
    axs[i].imshow(jnp.exp(log_prob.reshape(100, 100)), cmap=cm.viridis)
    for t in t_array:
      r_ = forward_fn(params, xi, jnp.ones(1) * t)
      axs[i].scatter(
        (r_[:, 0] + x_max) / 2 / x_max * 100,
        (r_[:, 2] + x_max) / 2 / x_max * 100,
        c="red",
        marker='.',
        s=.1
      )
    axs[i].axis("off")

  fig.tight_layout(pad=0.2)
  # plt.subplots_adjust(hspace=0.1)
  plt.savefig("results/fig/traj.pdf")


def plot_traj_and_velocity(
  sample_fn,
  forward_fn,
  inverse_fn,
  params: hk.Params,
  rng: PRNGKey,
  t_array: jnp.array,
):

  batch_size_pdf = 1024
  batch_size_velocity = 64
  fig1 = plt.figure(figsize=(10, 10))
  fig2 = plt.figure(figsize=(10, 10))
  ax1 = fig1.subplots(3, 2)
  ax2 = fig2.subplots(3, 2)
  i = 0
  for t in t_array:

    _rng, rng = jax.random.split(rng)
    fake_cond_ = np.ones((batch_size_pdf, 1)) * t
    samples = sample_fn(
      params, seed=_rng, sample_shape=(batch_size_pdf, ), cond=fake_cond_
    )
    ax1[i // 2, i % 2].scatter(samples[..., 0], samples[..., 1], s=1)

    fake_cond_ = np.ones((batch_size_velocity, 1)) * t
    x_min = -8
    x_max = 8
    x = np.linspace(x_min, x_max, 10)
    y = np.linspace(x_min, x_max, 10)
    X, Y = np.meshgrid(x, y)
    XY = jnp.hstack([X.reshape(10**2, 1), Y.reshape(10**2, 1)])
    # samples = sample_fn(params, seed=rng, sample_shape=(batch_size_velocity, ), cond=fake_cond_)
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
  plt.show()
