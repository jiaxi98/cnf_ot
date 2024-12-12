"""A simple example of a flow model trained to solve the regularized
Wasserstein proximal operator ."""
# TODO: code cleanup
# the key and rng are used alternatively throughout the code
# the name of the repo will be changed to cnf_ot
import pickle
from functools import partial
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import ml_collections
import optax
import yaml
from box import Box
from jaxtyping import Array
from tqdm import tqdm

import cnf_ot.utils as utils
from cnf_ot import applications
from cnf_ot.flows import RQSFlow
from cnf_ot.types import OptState, PRNGKey

jax.config.update("jax_enable_x64", True)


def main(config_dict: ml_collections.ConfigDict):

  # model initialization
  config = Box(config_dict)
  _type = config.general.type
  dim = config.general.dim
  dt = config.general.dt  # dt for calculating velocity using finite difference
  dx = config.general.dx  # dx for calculating score using finite difference
  t_batch_size = config.general.t_batch_size
  rng = jax.random.PRNGKey(config.general.seed)
  epochs = config.train.epochs
  batch_size = config.train.batch_size
  eval_frequency = config.train.eval_frequency
  _lambda = config.train._lambda

  model = RQSFlow(
    event_shape=(dim, ),
    num_layers=config.cnf.flow_num_layers,
    hidden_sizes=[config.cnf.hidden_size] * config.cnf.mlp_num_layers,
    num_bins=config.cnf.num_bins,
    periodized=False,
  )
  model = hk.without_apply_rng(hk.multi_transform(model))
  forward_fn = jax.jit(model.apply.forward)
  inverse_fn = jax.jit(model.apply.inverse)
  sample_fn = jax.jit(model.apply.sample, static_argnames=["sample_shape"])
  log_prob_fn = jax.jit(model.apply.log_prob)
  key, rng = jax.random.split(rng)
  params = model.init(key, jnp.zeros((1, dim)), jnp.zeros((1, )))
  optimizer = optax.adam(config.train.lr)
  opt_state = optimizer.init(params)

  if _type == "rwpo":
    T = config.rwpo.T
    beta = config.rwpo.beta
    a = config.rwpo.a
    subtype = config.rwpo.pot_type
    loss_fn = partial(
      applications.rwpo_loss_fn, model, dim, T, beta, dt, dx, t_batch_size,
      subtype
    )
    print(f"Solving regularized Wasserstein proximal in {dim}D...")
  elif _type == "fp":
    T = config.fp.T
    a = config.fp.a  # drift coeff
    sigma = config.fp.sigma
    subtype = config.fp.velocity_field_type
    loss_fn = partial(
      applications.fp_loss_fn, model, dim, T, a, sigma, dt, dx, t_batch_size,
      subtype
    )
  elif _type == "ot":
    T = 1
    subtype = config.ot.subtype
    loss_fn = partial(
      applications.ot_loss_fn, model, dim, T, dt, t_batch_size, subtype
    )
    print(f"Solving optimal transport in {dim}D...")
  else:
    raise Exception(f"Unknown problem type: {_type}...")

  @jax.jit
  def update(params: hk.Params, rng: PRNGKey, _lambda,
             opt_state: OptState) -> Tuple[Array, hk.Params, OptState]:
    """Single SGD update step."""
    loss, grads = jax.value_and_grad(loss_fn)(params, rng, _lambda, batch_size)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, new_opt_state

  # training loop
  loss_hist = []
  iters = tqdm(range(epochs))
  # jax.profiler.start_trace("runs")
  for step in iters:
    key, rng = jax.random.split(rng)
    loss, params, opt_state = update(params, key, _lambda, opt_state)
    #_lambda += density_fit_loss_fn(params, rng, _lambda, batch_size)
    loss_hist.append(loss)

    if step % eval_frequency == 0:
      desc_str = f"{loss=:.4e}"

      key, rng = jax.random.split(rng)
      if _type == "ot":
        KL = partial(applications.density_fit_rkl_loss_fn, model, dim,
                     T)(params, rng, batch_size)
        desc_str += f"{KL=:.4f}"
      # elif _type == "rwpo":
      #   # KL = reverse_kl_loss_fn(params, 0, rng, batch_size)
      #   KL = kl_loss_fn(params, 0, rng batch_size)
      #   pot = potential_loss_fn(params, T, rng, batch_size)
      #   kin = loss - KL * _lambda - pot
      #   desc_str += f"{KL=:.4f} | {pot=:.2f} | {kin=:.2f}"

      iters.set_description_str(desc_str)
  # jax.profiler.stop_trace()

  plt.plot(
    jnp.linspace(5001, epochs, epochs - 5000), jnp.array(loss_hist[5000:])
  )
  plt.savefig("results/fig/loss_hist.pdf")
  param_count = sum(x.size for x in jax.tree.leaves(params))
  print("Network parameters: {}".format(param_count))

  if _type == "ot":
    print(
      "kinetic energy with more samples: ",
      utils.calc_kinetic_energy(
        sample_fn, params, rng, batch_size=65536, t_size=10000, dim=dim
      )
    )

    print(
      "kinetic energy with less samples: ",
      utils.calc_kinetic_energy(
        sample_fn, params, rng, batch_size=4096, t_size=1000, dim=dim
      )
    )
  elif _type == "rwpo":
    e_kin = T * utils.calc_score_kinetic_energy(
      sample_fn,
      log_prob_fn,
      params,
      T,
      beta,
      dim,
      rng,
    )
    e_pot = partial(applications.potential_loss_fn, model, dim, a,
                    subtype)(params, T, rng, 65536)
    print("kinetic energy: ", e_kin)
    print("potential energy: ", e_pot)

    if subtype == "quadratic":
      # NOTE: this is the true value for quadratic potential and Gaussian IC
      true_val = dim * (1 + jnp.log(T + 1)) / beta
    elif subtype == "double-well":
      if a == 0.5:
        file_name = 'data/fcn4a5_interp.pkl'
      elif a == 1.0:
        file_name = 'data/fcn4a1_interp.pkl'
      with open(file_name, 'rb') as f:
        interpolators = pickle.load(f)
        target_prob = interpolators['rhoT_interp']

      x_min = -2
      x_max = 2
      x = jnp.linspace(x_min, x_max, 100)
      y = jnp.linspace(x_min, x_max, 100)
      X, Y = jnp.meshgrid(x, y)
      XY = jnp.hstack([X.reshape(100**2, 1), Y.reshape(100**2, 1)])

      def cost_rwpo(rng: PRNGKey, x_batch: int, y_batch: int):
        """calculate the cost of the rwpo"""

        rng, key = jax.random.split(rng)
        x = jax.random.normal(rng, shape=(x_batch, 2)) *\
          jnp.sqrt(2 / beta * (T + 1))
        y = jax.random.normal(key, shape=(x_batch, y_batch, 2)) *\
          jnp.sqrt(2 / beta * T) + x.reshape(-1, 1, 2)
        return -2 / beta * jnp.log(
          jnp.exp(
            applications.potential_fn(
              y.reshape(-1, 2)).reshape((x_batch, y_batch)
            ) *\
            -beta / 2
          ).mean(axis=1)
        ).mean()

      fake_cond_ = jnp.ones((1, ))
      prob1 = jnp.exp(log_prob_fn(params, XY, cond=fake_cond_))
      prob2 = target_prob(XY)
      print(jnp.sum((prob1 - prob2)**2))
      plt.figure(figsize=(4, 2))
      plt.subplot(121)
      plt.imshow(prob1.reshape(100, 100))
      plt.subplot(122)
      plt.imshow(prob2.reshape(100, 100))
      plt.savefig("results/fig/double-well.pdf")
      true_val = cost_rwpo(rng, 100, 1000)
    print(
      "total energy: {:.3f}|relative err: {:.3e}".format(
        e_kin + e_pot, (e_kin + e_pot - true_val) / true_val * 100
      )
    )
  elif _type == "fp":
    source_prob = jax.vmap(
      partial(
        jax.scipy.stats.multivariate_normal.pdf,
        mean=jnp.zeros(dim),
        cov=4 * jnp.eye(dim)
      )
    )
    target_prob = jax.vmap(
      partial(
        jax.scipy.stats.multivariate_normal.pdf,
        mean=jnp.zeros(dim),
        cov=jnp.eye(dim) *
        (jnp.exp(-2 * a * T) * (4 - 1 / 2 / a) + 1 / 2 / a),
      )
    )
    def rmse_mc_loss_fn(
      params: hk.Params, cond: float, rng: PRNGKey, batch_size: int
    ) -> Array:
      """MSE between the normalizing flow and the reference distribution.
      """

      fake_cond_ = jnp.ones((batch_size, 1)) * cond
      samples, log_prob = model.apply.sample_and_log_prob(
        params,
        cond=fake_cond_,
        seed=rng,
        sample_shape=(batch_size, ),
      )
      return jnp.sqrt(
        (
          (
            jnp.exp(log_prob) -
            (source_prob(samples) * (1 - cond) + target_prob(samples) * cond)
          )**2
        ).mean()
      )

    print(
      "L2 error via Monte-Carlo: {:.3e}".format(
        rmse_mc_loss_fn(params, 1, rng, 1000000)
      )
    )

    if dim == 2:
      def rmse_grid_loss_fn(params: hk.Params, cond, grid_size: int) -> Array:
        """MSE between the normalizing flow and the reference distribution.
        """

        fake_cond_ = jnp.ones(1) * cond
        x_min = -5
        x_max = 5
        x = jnp.linspace(x_min, x_max, grid_size)
        y = jnp.linspace(x_min, x_max, grid_size)
        X, Y = jnp.meshgrid(x, y)
        XY = jnp.hstack([X.reshape(-1, 1), Y.reshape(-1, 1)])
        return jnp.sqrt(
          (
            (
              jnp.exp(log_prob_fn(params, XY, fake_cond_)) -
              (source_prob(XY) * (1 - cond) + target_prob(XY) * cond)
            )**2
          ).mean()
        )

      r_ = jnp.vstack(
        [
          jnp.array([-1.0, -1.0]),
          jnp.array([-1.0, -0.0]),
          jnp.array([-1.0, 1.0]),
          jnp.array([0.0, -1.0]),
          jnp.array([0.0, 0.0]),
          jnp.array([0.0, 1.0]),
          jnp.array([1.0, -1.0]),
          jnp.array([1.0, 0.0]),
          jnp.array([1.0, 1.0])
        ]
      )
      r_ = r_ * 3
      t_array = jnp.linspace(0, T, 20)
      utils.plot_density_and_trajectory(
        forward_fn,
        inverse_fn,
        log_prob_fn,
        params=params,
        r_=r_,
        t_array=t_array,
      )
      print("L2 error on grid: {:.3e}".format(rmse_grid_loss_fn(params, 1, 500)))
  
  breakpoint()

  if dim == 2:
    # this plot the distribution at t=0,1 after training
    # as well as the error of the learned mapping at t=0, 1
    # based on grid evaluation

    r_ = jnp.vstack(
      [
        jnp.array([-1.0, -1.0]),
        jnp.array([-1.0, -0.0]),
        jnp.array([-1.0, 1.0]),
        jnp.array([0.0, -1.0]),
        jnp.array([0.0, 0.0]),
        jnp.array([0.0, 1.0]),
        jnp.array([1.0, -1.0]),
        jnp.array([1.0, 0.0]),
        jnp.array([1.0, 1.0])
      ]
    )
    r_ = r_ * 3
    # utils.plot_score(log_prob_fn, params, r_)
    # breakpoint()

    t_array = jnp.linspace(0, T, 20)
    utils.plot_density_and_trajectory(
      forward_fn,
      inverse_fn,
      log_prob_fn,
      params=params,
      r_=r_,
      t_array=t_array,
    )

  # plot the 1D rwpo example：
  # plot the histogram w.r.t. the ground truth solution
  # plot the velocity at several time step v.s. the ground truth
  # if _type == "rwpo" and dim == 1:
  #   plt.clf()
  #   t = jnp.linspace(0, 1, 6)
  #   for i in range(2):
  #     for j in range(3):
  #       plt.subplot(2, 3, i * 3 + j + 1)
  #       fake_cond = jnp.ones((test_batch_size, 1)) * t[i * 3 + j]
  #       samples = sample_fn(
  #         params,
  #         seed=key,
  #         sample_shape=(test_batch_size, ),
  #         cond=fake_cond
  #       )
  #       plt.hist(samples[..., 0], bins=bins * 4, density=True)
  #       x = jnp.linspace(-5, 5, 1000)
  #       rho = jax.vmap(
  #         distrax.Normal(loc=0,
  #                        scale=jnp.sqrt(beta * 2 * (2 - t[i * 3 + j]))).prob
  #       )(x)
  #       plt.plot(x, rho, label=r"$\rho_*$")
  #       plt.legend()
  #   plt.savefig("results/fig/rwpo.pdf")
  #   plt.clf()

  #   kinetic_err = []
  #   t_array = jnp.linspace(0, 1, 101)
  #   batch_size = 1000
  #   for t in t_array:
  #     fake_cond_ = jnp.ones((batch_size, 1)) * t
  #     samples = sample_fn(
  #       params, seed=rng, sample_shape=(batch_size, ), cond=fake_cond_
  #     )
  #     xi = inverse_fn(params, samples, fake_cond_)
  #     velocity = jax.jacfwd(partial(forward_fn, params,
  #                                   xi))(fake_cond_)[jnp.arange(batch_size), :,
  #                                                    jnp.arange(batch_size), 0]
  #     ground_truth = -jnp.sqrt(1 / 8 / (2 - t)) * samples
  #     kinetic_err.append(jnp.mean((velocity - ground_truth)**2))
  #     plt.figure(figsize=(4, 2))
  #     plt.scatter(samples, velocity, c="b", label="compute", s=.1)
  #     plt.scatter(samples, ground_truth, c="r", label="ground truth", s=.1)
  #     plt.legend()
  #     plt.title("t = {:.2f}".format(t))
  #     plt.savefig("results/fig/{:.2f}.pdf".format(t))
  #     plt.clf()
  #   plt.plot(
  #     t_array, kinetic_err, label=r"$\left\| \dot{x} - \dot{x}_* \right\|^2$"
  #   )
  #   plt.legend()
  #   plt.savefig("results/fig/rwpo_kin.pdf")
  #   breakpoint()

  #   batch_size = batch_size
  #   loss = potential_loss_fn(params, 1, rng, batch_size)
  #   t_batch_size = 100  # 10
  #   t_batch = jax.random.uniform(rng, (t_batch_size, ))
  #   for _ in range(t_batch_size):
  #     loss += kinetic_loss_fn(
  #       t_batch[_], params, rng, batch_size // 32
  #     ) / t_batch_size
  #   print("loss: {:.4f}".format(loss))


if __name__ == "__main__":

  with open("config/main.yaml", "r") as file:
    config_dict = yaml.safe_load(file)
  main(config_dict)
