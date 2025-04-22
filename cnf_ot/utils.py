import re
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.colors as mcolors
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from cnf_ot.types import PRNGKey


###############################################################################
# visualizations for unconditional normalizing flows
###############################################################################
def plot_dim_reduction_reconst(
  forward_fn: callable,
  inverse_fn: callable,
  params_1: hk.Params,
  params_2: hk.Params,
  dim: int,
  sub_dim: int,
  samples: jnp.ndarray,
):

  transf = forward_fn(params_1, samples)
  transf = transf.at[:, sub_dim:].set(0)
  reconst = inverse_fn(params_2, transf)
  if dim == 2:
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    axs[0].scatter(samples[..., 0], samples[..., 1], s=1, c=samples[..., 0])
    axs[0].set_title("original")
    axs[1].scatter(transf[..., 0], transf[..., 1], s=1, c=samples[..., 0])
    axs[1].set_title("transformed")
    axs[2].scatter(reconst[..., 0], reconst[..., 1], s=1, c=samples[..., 0])
    axs[2].set_title("reconstructed")
  elif dim == 3:
    fig = plt.figure(figsize=(9, 3))
    ax = fig.add_subplot(131, projection='3d')
    ax.scatter(
      samples[..., 0], samples[..., 1], samples[..., 2], s=1, c=samples[..., 0]
    )
    ax.set_title("original")
    ax.view_init(elev=40, azim=45)
    ax = fig.add_subplot(132, projection='3d')
    ax.scatter(
      transf[..., 0], transf[..., 1], transf[..., 2], s=1, c=samples[..., 0]
    )
    ax.set_title("transformed")
    ax = fig.add_subplot(133, projection='3d')
    ax.scatter(
      reconst[..., 0], reconst[..., 1], reconst[..., 2], s=1, c=samples[..., 0]
    )
    ax.set_title("reconstructed")
    ax.view_init(elev=40, azim=45)
  fig.tight_layout()
  plt.savefig("results/fig/dr.png")
  plt.clf()


def plot_samples_snapshot(
  sample_fn: callable,
  params: hk.Params,
  rng: PRNGKey,
  batch_size,
):

  samples = sample_fn(params, seed=rng, sample_shape=(batch_size, ))
  plt.scatter(
    samples[..., 0],
    samples[..., 1],
    s=1,
  )
  plt.savefig("results/fig/samples.png")
  plt.clf()


def plot_density_snapshot(
  log_prob_fn: callable,
  params: hk.Params,
):

  x_min = -6
  x_max = 6
  x = np.linspace(x_min, x_max, 100)
  y = np.linspace(x_min, x_max, 100)
  X, Y = np.meshgrid(x, y)
  XY = jnp.hstack([X.reshape(100**2, 1), Y.reshape(100**2, 1)])
  log_prob = log_prob_fn(params, XY)
  plt.imshow(jnp.exp(log_prob.reshape(100, 100)))
  plt.axis("off")

  plt.savefig("results/fig/density.png")
  plt.clf()


def plot_dimension_reduction():
  # this color is only for visualizing ordered samples, e.g. unit circle
  assert dim <= 3
  if config.type == "S1":
    color = random.uniform(rng, (batch_size, ))
    data = data.at[:, 0].set(jnp.sin(2 * jnp.pi * color))
    data = data.at[:, 1].set(jnp.cos(2 * jnp.pi * color))
  if model == "enc_dec":
    utils.plot_dim_reduction_reconst(
      encoder_forward_fn,
      decoder_forward_fn,
      params1["encoder"],
      params1["decoder"],
      dim,
      sub_dim,
      data,
    )
  elif model == "dec_only":
    utils.plot_dim_reduction_reconst(
      decoder_inverse_fn,
      decoder_forward_fn,
      params1,
      params1,
      dim,
      sub_dim,
      data,
    )


def find_mfd_path(
  encoders, decoders, params, data1, data2, overlap, sub_dim, start, end,
  fig_name
):

  path_length = 100
  mid = overlap[0]
  t = jnp.linspace(0, 1, path_length)

  start_coord = encoders[0].apply.forward(params[0]["encoder"], start)
  mid_coord = encoders[0].apply.forward(params[0]["encoder"], mid)
  path1_coord = start_coord + t[:, None] * (mid_coord - start_coord)
  path1_coord = path1_coord.at[:, sub_dim:].set(0)
  path1 = decoders[0].apply.forward(params[0]["decoder"], path1_coord)

  mid_coord = encoders[1].apply.forward(params[1]["encoder"], mid)
  end_coord = encoders[1].apply.forward(params[1]["encoder"], end)
  path2_coord = mid_coord + t[:, None] * (end_coord - mid_coord)
  path2_coord = path2_coord.at[:, sub_dim:].set(0)
  path2 = decoders[1].apply.forward(params[1]["decoder"], path2_coord)
  path = jnp.concatenate([path1, path2], axis=0)

  fig = plt.figure(figsize=(6, 6))
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(data1[..., 0], data1[..., 1], data1[..., 2], s=1, c='red')
  ax.scatter(data2[..., 0], data2[..., 1], data2[..., 2], s=1, c='blue')
  ax.scatter(path[..., 0], path[..., 1], path[..., 2], s=1, c='black')
  ax.scatter(start[0], start[1], start[2], s=30, c='yellow')
  ax.scatter(mid[0], mid[1], mid[2], s=30, c='yellow')
  ax.scatter(end[0], end[1], end[2], s=30, c='yellow')
  ax.view_init(elev=10, azim=45)
  plt.savefig(f"results/fig/{fig_name}", dpi=500)


def find_long_mfd_path(
  encoders, decoders, params, charts, pos, radius, sub_dim, start, end, data,
  fig_name
):

  path_length = 100
  x0 = start
  t = jnp.linspace(0, 1, path_length)
  path = start[None]

  cluster_colors = np.linspace(0, 1, len(charts))
  cmap = LinearSegmentedColormap.from_list('RedToBlue', ['red', 'blue'])
  fig = plt.figure(figsize=(6, 6))
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(x0[0], x0[1], x0[2], s=10, c='yellow')

  for i in range(len(charts) - 1):
    center = pos[i + 1]
    x1 = charts[i][jnp.linalg.norm(charts[i] - center, axis=-1) < radius[i +
                                                                         1]][0]
    x0_coord = encoders[i].apply.forward(params[i]["encoder"], x0)
    x1_coord = encoders[i].apply.forward(params[i]["encoder"], x1)
    path_coord = x0_coord + t[:, None] * (x1_coord - x0_coord)
    path_coord = path_coord.at[:, sub_dim:].set(0)
    path_ = decoders[i].apply.forward(params[i]["decoder"], path_coord)
    path = jnp.concatenate([path, path_], axis=0)

    ax.scatter(x1[0], x1[1], x1[2], s=30, c='yellow')
    ax.scatter(
      charts[i][..., 0],
      charts[i][..., 1],
      charts[i][..., 2],
      s=1,
      c=cmap(cluster_colors[i])
    )
    ax.scatter(path_[..., 0], path_[..., 1], path_[..., 2], s=1, c='black')
    x0 = x1

  i = -1
  x1 = end
  x0_coord = encoders[i].apply.forward(params[i]["encoder"], x0)
  x1_coord = encoders[i].apply.forward(params[i]["encoder"], x1)
  path_coord = x0_coord + t[:, None] * (x1_coord - x0_coord)
  path_coord = path_coord.at[:, sub_dim:].set(0)
  path_ = decoders[i].apply.forward(params[i]["decoder"], path_coord)
  path = jnp.concatenate([path, path_], axis=0)

  ax.scatter(x1[0], x1[1], x1[2], s=30, c='yellow')
  ax.scatter(
    charts[i][..., 0], charts[i][..., 1], charts[i][..., 2], s=1, c="blue"
  )
  ax.scatter(path_[..., 0], path_[..., 1], path_[..., 2], s=1, c='black')
  ax.scatter(data[..., 0], data[..., 1], data[..., 2], s=1, c='red', alpha=0.1)
  ax.view_init(elev=10, azim=45)
  plt.savefig(f"results/fig/{fig_name}", dpi=500)

  return path


def check_path_accuracy(path, type_):
  """check the accuracy of the path."""

  if type_[0] == "S":
    return jnp.mean(jnp.abs(jnp.sum(path**2, axis=-1) - 1))
  elif type_[0] == "T":
    R = 5
    r = 1
    tmp = jnp.sqrt(path[..., 0]**2 + path[..., 1]**2)
    return jnp.mean(jnp.abs((tmp - R)**2 + path[..., 2]**2 - r**2))


###############################################################################
# visualizations for conditional normalizing flows
###############################################################################
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
    y0 = 8 / 5
    x0 = 3 / 2
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
    field = -jnp.concat(
      [(dpidx / _pi)[:, None], (dpidy / _pi)[:, None]], axis=1
    )
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
  domain_range: list,
):

  plt.clf()
  fig, axs = plt.subplots(
    t_array.shape[0] // 5, 5, figsize=(5, t_array.shape[0] // 5)
  )
  axs = axs.flatten()
  x_min, x_max, y_min, y_max = domain_range[0], domain_range[1],\
    domain_range[2], domain_range[3]
  x = np.linspace(x_min, x_max, 100)
  y = np.linspace(y_min, y_max, 100)
  X, Y = np.meshgrid(x, y)
  XY = jnp.hstack([X.reshape(100**2, 1), Y.reshape(100**2, 1)])
  xi = inverse_fn(params, r_, jnp.zeros(1))

  for i in range(len(t_array)):
    index = 20
    fake_cond_ = t_array[i] * jnp.ones((1, ))
    log_prob = log_prob_fn(params, XY, cond=fake_cond_)
    axs[i].imshow(jnp.exp(log_prob.reshape(100, 100)), cmap="Greens")
    for t in t_array:
      r_ = forward_fn(params, xi, jnp.ones(1) * t)
      axs[i].scatter(
        (r_[:, 0] - x_min) / (x_max - x_min) * 100,
        (r_[:, 1] - y_min) / (y_max - y_min) * 100,
        c="red",
        s=.1 * index
      )
      index -= 4.75
    axs[i].axis("off")
    axs[i].set_xlabel("x")
    axs[i].set_xlabel("y")
    axs[i].set_title("t = {:.2f}".format(t_array[i]), fontsize=5, y=-0.2)

  fig.tight_layout()
  # plt.subplots_adjust(hspace=0.1)
  plt.savefig("results/fig/traj.pdf")


def plot_high_dim_density_and_trajectory(
  forward_fn: callable,
  inverse_fn: callable,
  log_prob_fn: callable,
  params: hk.Params,
  r_: jnp.array,
  t_array: jnp.array,
  domain_range: list,
):

  plt.clf()
  fig, axs = plt.subplots(
    t_array.shape[0] // 5, 5, figsize=(5, t_array.shape[0] // 5)
  )
  axs = axs.flatten()
  x_min, x_max, y_min, y_max = domain_range[0], domain_range[1],\
    domain_range[2], domain_range[3]
  x = np.linspace(x_min, x_max, 100)
  y = np.linspace(y_min, y_max, 100)
  X, Y = np.meshgrid(x, y)
  XY = jnp.hstack(
    [X.reshape(100**2, 1),
     Y.reshape(100**2, 1),
     jnp.ones((100**2, 1)) * 3]
  )
  xi = inverse_fn(params, r_, jnp.zeros(1))

  for i in range(len(t_array)):
    index = 20
    fake_cond_ = t_array[i] * jnp.ones((1, ))
    log_prob = log_prob_fn(params, XY, cond=fake_cond_)
    axs[i].imshow(
      jnp.exp((log_prob.reshape(100, 100)).T)[:, ::-1], cmap="Greens"
    )
    for t in t_array:
      r_ = forward_fn(params, xi, jnp.ones(1) * t)
      axs[i].scatter(
        (r_[:, 0] - x_min) / (x_max - x_min) * 100,
        (r_[:, 1] - y_min) / (y_max - y_min) * 100,
        c="red",
        s=.1 * index
      )
      index -= 2
    axs[i].axis("off")
    axs[i].set_title("t = {:.2f}".format(t_array[i]), fontsize=5, y=-0.2)

  fig.tight_layout()
  plt.savefig("results/fig/traj.pdf")


def plot_proj_density(
  log_prob_fn: callable,
  params: hk.Params,
  t_array: jnp.array,
  domain_range: list,
  direction: str = 'z',
):

  plt.clf()
  fig, axs = plt.subplots(
    t_array.shape[0] // 5, 5, figsize=(5, t_array.shape[0] // 5)
  )
  axs = axs.flatten()
  x_min, x_max, y_min, y_max = domain_range[0], domain_range[1],\
    domain_range[2], domain_range[3]
  N = 100
  x = np.linspace(x_min, x_max, N)
  y = np.linspace(y_min, y_max, N)
  X, Y = np.meshgrid(x, y)
  section = np.linspace(-5, 5, 11)

  for j in range(len(t_array)):
    prob = np.zeros((N, N))
    for i in range(len(section)):
      if direction == 'x':
        XYZ = jnp.hstack(
          [
            jnp.ones((N**2, 1)) * section[i],
            X.reshape(N**2, 1),
            Y.reshape(N**2, 1)
          ]
        )
      elif direction == 'y':
        XYZ = jnp.hstack(
          [
            X.reshape(N**2, 1),
            jnp.ones((N**2, 1)) * section[i],
            Y.reshape(N**2, 1)
          ]
        )
      elif direction == 'z':
        XYZ = jnp.hstack(
          [
            X.reshape(N**2, 1),
            Y.reshape(N**2, 1),
            jnp.ones((N**2, 1)) * section[i]
          ]
        )
      fake_cond_ = t_array[j] * jnp.ones((1, ))
      prob += np.exp(log_prob_fn(params, XYZ, cond=fake_cond_)).reshape(N, N)
    prob /= len(section)
    axs[j].imshow((prob.T)[:, ::-1], cmap="Greens")
    axs[j].axis("off")
    axs[j].set_title("t = {:.2f}".format(t_array[j]), fontsize=5, y=-0.2)

  fig.tight_layout()
  plt.savefig(f"results/fig/proj_density_{direction}.pdf")


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


def sympy_to_latex(expr_str):
  # / to \frac
  expr_str = re.sub(r'\(([^/]+)\)/([^/+*-]+)', r'\\frac{\1}{\2}', expr_str)
  # ** to ^
  expr_str = re.sub(r'(\w+)\*\*(\w+)', r'\1^{\2}', expr_str)
  # math symbol
  replacements = {
    r'\bdelta\b': r'\\delta',
    r'\blog\b': r'\\log',
    r'\*': '',
  }

  for pattern, repl in replacements.items():
    expr_str = re.sub(pattern, repl, expr_str)

  return f"$$\n{expr_str}\n$$"
