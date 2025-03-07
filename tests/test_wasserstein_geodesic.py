"""A simple example of a flow model trained to solve the Wassserstein geodesic problem."""
from functools import partial
from typing import Iterator, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import distrax
import matplotlib.pyplot as plt
import numpy as np
import optax
import tensorflow_datasets as tfds
from absl import app, flags, logging
from jaxtyping import Array
from tqdm import tqdm

from src.flows import RQSFlow
from src.types import Batch, OptState, PRNGKey
import src.utils as utils

flags.DEFINE_integer(
  "flow_num_layers", 1, "Number of layers to use in the flow."
)
flags.DEFINE_integer(
  "mlp_num_layers", 2, "Number of layers to use in the MLP conditioner."
)  # 2
flags.DEFINE_integer(
  "hidden_size", 16, "Hidden size of the MLP conditioner."
)  # 64
flags.DEFINE_integer(
  "num_bins", 20, "Number of bins to use in the rational-quadratic spline."
)  # 20
flags.DEFINE_integer("batch_size", 2048, "Batch size for training.")  #2048
flags.DEFINE_integer(
  "test_batch_size", 2000, "Batch size for evaluation."
)  #20000
flags.DEFINE_float("lr", 1e-3, "Learning rate for the optimizer.")
flags.DEFINE_integer("epochs", 5000, "Number of training steps to run.")
flags.DEFINE_integer("eval_frequency", 100, "How often to evaluate the model.")
flags.DEFINE_integer("seed", 42, "random seed.")

flags.DEFINE_enum(
  "target", "unimodal", ["unimodal", "multimodal", "correlated"],
  "which target distribution to use"
)

flags.DEFINE_boolean('use_64', True, 'whether to use float64')
flags.DEFINE_boolean('plot', False, 'whether to plot resulting model density')

flags.DEFINE_integer("dim", 1, "dimension of the base space")

FLAGS = flags.FLAGS

# def gaussian_distribution_sampler(
#     prng_key: int=42,
#     mean: jnp.ndarray=0,
#     var: jnp.ndarray=1,
#     batch_size: int=256
#     ) -> jnp.ndarray:
#     """
#     TODO: test the sampler
#     """

#     dim = mean.shape[0]
#     C = jnp.linalg.cholesky(var)
#     sample = jax.random.normal(jax.random.PRNGKey(prng_key), (batch_size, dim)) @ C.T + jnp.reshape(mean, (1, dim))
#     return sample


def gaussian_2d(
  r: jnp.ndarray,
  mean: jnp.ndarray = jnp.array([2, 3]),
  var: jnp.ndarray = jnp.array([[2, .5], [.5, 1]])
) -> jnp.ndarray:

  return jnp.exp(
    -0.5 * jnp.dot(jnp.dot((r - mean), jnp.linalg.inv(var)), (r - mean).T)
  ) / jnp.sqrt(jnp.linalg.det(2 * jnp.pi * var))


def potential_fn(r: jnp.ndarray, ) -> jnp.ndarray:
  return jnp.sum(r**2, axis=1) / 2


def main(_):

  # model initialization
  jax.config.update("jax_enable_x64", FLAGS.use_64)
  np.random.seed(FLAGS.seed)
  rng = jax.random.PRNGKey(FLAGS.seed)
  optimizer = optax.adam(FLAGS.lr)

  model = RQSFlow(
    event_shape=(FLAGS.dim, ),
    num_layers=FLAGS.flow_num_layers,
    hidden_sizes=[FLAGS.hidden_size] * FLAGS.mlp_num_layers,
    num_bins=FLAGS.num_bins,
    periodized=False,
  )
  model = hk.without_apply_rng(hk.multi_transform(model))
  forward_fn = jax.jit(model.apply.forward)
  inverse_fn = jax.jit(model.apply.inverse)
  sample_fn = jax.jit(model.apply.sample, static_argnames=['sample_shape'])
  key, rng = jax.random.split(rng)
  params = model.init(key, np.zeros((1, FLAGS.dim)), np.zeros((1, 1)))
  print(params.keys())

  opt_state = optimizer.init(params)
  bins = 25

  # boundary condition on density
  if FLAGS.dim == 1:
    # 1D case

    # Gaussian source and target
    # source_prob = jax.vmap(distrax.Normal(loc=0, scale=1).prob)
    # target_prob = jax.vmap(distrax.Normal(loc=3, scale=1).prob)

    def source_prob_(r):
      return distrax.Normal(loc=0, scale=1).prob(r) * .3 + distrax.Normal(
        loc=4, scale=1
      ).prob(r) * .7

    source_prob = jax.vmap(source_prob_)
    target_prob = jax.vmap(distrax.Normal(loc=2, scale=1).prob)

  elif FLAGS.dim == 2:
    # 2D case
    source_prob = jax.vmap(
      partial(gaussian_2d, mean=jnp.array([-1, -1]), var=jnp.eye(2))
    )
    target_prob = jax.vmap(
      partial(gaussian_2d, mean=jnp.array([3, 3]), var=jnp.eye(2))
    )

  @partial(jax.jit, static_argnames=['batch_size'])
  def w_loss_fn(params: hk.Params, rng: PRNGKey, batch_size: int) -> Array:
    """Loss of the Wasserstein gradient flow, including:

    KL divergence between the normalizing flow at t=0 and the source distribution
    KL divergence between the normalizing flow at t=1 and the target distribution
    Monte-Carlo integration of the kinetic energy along the interval [0, 1]

    NOTE: adding loss function corresponds to acc will significantly slower the computation

    TODO: one caveat of the coding here is that we do not further split the 
    rng for sampling from CNF in kl_loss and kinetic_loss. 
    """

    def double_kl_loss_fn(
      params: hk.Params, rng: PRNGKey, batch_size: int
    ) -> Array:

      prob_fn = source_prob
      loss = kl_loss_fn(params, rng, 0, batch_size)
      prob_fn = target_prob
      loss += kl_loss_fn(params, rng, 1, batch_size)

      @partial(jax.jit, static_argnames=['batch_size'])
      def kl_loss_fn(
        params: hk.Params, rng: PRNGKey, cond, batch_size: int
      ) -> Array:
        """KL-divergence between the normalizing flow and the reference distribution.
        
        TODO: here, we assume the p.d.f. of the target distribution is known. 
        In the case where we only access to samples from target distribution,
        KL-divergence is not calculable and we need to shift to other integral 
        probability metric, e.g. MMD.
        """

        breakpoint()
        fake_cond_ = np.ones((batch_size, 1)) * cond
        samples, log_prob = model.apply.sample_and_log_prob(
          params,
          cond=fake_cond_,
          seed=rng,
          sample_shape=(batch_size, ),
        )
        return (log_prob - jnp.log(prob_fn(samples))).mean()

    @partial(jax.jit, static_argnames=['batch_size'])
    def kinetic_loss_fn(
      t: float, params: hk.Params, rng: PRNGKey, batch_size: int
    ) -> Array:
      """Kinetic energy along the trajectory at time t
      """
      fake_cond_ = np.ones((batch_size, 1)) * t
      samples = sample_fn(
        params, seed=rng, sample_shape=(batch_size, ), cond=fake_cond_
      )
      xi = inverse_fn(params, samples, fake_cond_)
      velocity = jax.jacfwd(partial(forward_fn, params, xi))(fake_cond_)
      # velocity.shape = [batch_size, 2, batch_size, 1]
      # velocity[jnp.arange(batch_size),:,jnp.arange(batch_size),0].shape = [batch_size, 2]
      weight = .01
      return jnp.mean(
        velocity[jnp.arange(batch_size), :,
                 jnp.arange(batch_size), 0]**2
      ) * weight * FLAGS.dim / 2

    @partial(jax.jit, static_argnames=['batch_size'])
    def acc_loss_fn(
      t: float, params: hk.Params, rng: PRNGKey, batch_size: int
    ) -> Array:
      """acceleration energy along the trajectory at time t, used for regularization
      """
      fake_cond_ = np.ones((batch_size, 1)) * t
      samples = sample_fn(
        params, seed=rng, sample_shape=(batch_size, ), cond=fake_cond_
      )
      xi = inverse_fn(params, samples, fake_cond_)
      velocity = jax.jacfwd(partial(forward_fn, params, xi))(fake_cond_)
      # velocity.shape = [batch_size, 2, batch_size, 1]
      # velocity[jnp.arange(batch_size),:,jnp.arange(batch_size),0].shape = [batch_size, 2]

      dt = 0.01
      fake_cond_ = np.ones((batch_size, 1)) * (t + dt)
      samples = sample_fn(
        params, seed=rng, sample_shape=(batch_size, ), cond=fake_cond_
      )
      xi = inverse_fn(params, samples, fake_cond_)
      velocity_ = jax.jacfwd(partial(forward_fn, params, xi))(fake_cond_)
      #weight = .01
      return jnp.mean(
        velocity[jnp.arange(batch_size), :,
                 jnp.arange(batch_size), 0] -
        velocity_[jnp.arange(batch_size), :,
                  jnp.arange(batch_size), 0]
      ) * FLAGS.dim / 2

    loss = double_kl_loss_fn(params, rng, batch_size)
    t_batch_size = 10  # 10
    t_batch = jax.random.uniform(rng, (t_batch_size, ))
    for _ in range(t_batch_size):
      loss += kinetic_loss_fn(t_batch[_], params, rng,
                              batch_size // 32) / t_batch_size + acc_loss_fn(
                                t_batch[_], params, rng, batch_size // 32
                              ) / t_batch_size

    return loss

  @partial(jax.jit, static_argnames=['batch_size'])
  def mfg_loss_fn(params: hk.Params, rng: PRNGKey, batch_size: int) -> Array:
    '''Loss of the mean-field potential game
    '''

    def potential_loss_fn(
      params: hk.Params, rng: PRNGKey, cond, potential_fn, batch_size: int
    ) -> Array:

      fake_cond_ = np.ones((batch_size, 1)) * cond
      samples, _ = model.apply.sample_and_log_prob(
        params,
        cond=fake_cond_,
        seed=rng,
        sample_shape=(batch_size, ),
      )
      return potential_fn(samples).mean()

    def kl_loss_fn(
      params: hk.Params, rng: PRNGKey, cond, batch_size: int
    ) -> Array:

      fake_cond_ = np.ones((batch_size, 1)) * cond
      samples, log_prob = model.apply.sample_and_log_prob(
        params,
        cond=fake_cond_,
        seed=rng,
        sample_shape=(batch_size, ),
      )
      return (log_prob - jnp.log(prob_fn(samples))).mean()

    prob_fn = source_prob
    loss = kl_loss_fn(params, rng, 0, batch_size)
    loss += potential_loss_fn(params, rng, 1, potential_fn, batch_size)
    return loss

  @jax.jit
  def update(params: hk.Params, rng: PRNGKey,
             opt_state: OptState) -> Tuple[Array, hk.Params, OptState]:
    """Single SGD update step."""
    loss, grads = jax.value_and_grad(mfg_loss_fn)(params, rng, FLAGS.batch_size)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, new_opt_state

  # plot the distribution at t=0, 1 before training
  plt.subplot(121)
  if FLAGS.dim == 1:
    fake_cond = np.zeros((FLAGS.batch_size, 1))
    samples = sample_fn(
      params, seed=key, sample_shape=(FLAGS.batch_size, ), cond=fake_cond
    )
    plt.hist(samples[..., 0], bins=bins * 4, density=True)
    fake_cond = np.ones((FLAGS.batch_size, 1))
    samples = sample_fn(
      params, seed=key, sample_shape=(FLAGS.batch_size, ), cond=fake_cond
    )
    plt.hist(samples[..., 0], bins=bins * 4, density=True)
  elif FLAGS.dim == 2:
    fake_cond = np.zeros((FLAGS.batch_size, 1))
    samples = sample_fn(
      params, seed=key, sample_shape=(FLAGS.batch_size, ), cond=fake_cond
    )
    plt.scatter(samples[..., 0], samples[..., 1], s=3, c='r')
    fake_cond = np.ones((FLAGS.batch_size, 1))
    samples = sample_fn(
      params, seed=key, sample_shape=(FLAGS.batch_size, ), cond=fake_cond
    )
    plt.scatter(samples[..., 0], samples[..., 1], s=1, c='b')
  # training loop
  loss_hist = []
  iters = tqdm(range(FLAGS.epochs))
  for step in iters:
    key, rng = jax.random.split(rng)
    loss, params, opt_state = update(params, key, opt_state)
    loss_hist.append(loss)

    # if step % FLAGS.eval_frequency == 0:
    #   desc_str = f"{loss=:.2f}"

    #   key, rng = jax.random.split(rng)
    #   KL = kl_loss_fn(params, rng, FLAGS.batch_size)
    #   kin = kinetic_loss_fn(0.5, params, rng, FLAGS.batch_size)
    #   desc_str += f" | {KL=:.2f} | {kin=:.2f}"
    #   iters.set_description_str(desc_str)

  # plot the distribution at t=0, 1 after training
  plt.subplot(122)
  if FLAGS.dim == 1:
    fake_cond = np.zeros((FLAGS.batch_size, 1))
    samples = sample_fn(
      params, seed=key, sample_shape=(FLAGS.batch_size, ), cond=fake_cond
    )
    plt.hist(samples[..., 0], bins=bins * 4, density=True)
    fake_cond = np.ones((FLAGS.batch_size, 1))
    samples = sample_fn(
      params, seed=key, sample_shape=(FLAGS.batch_size, ), cond=fake_cond
    )
    plt.hist(samples[..., 0], bins=bins * 4, density=True)
    print(
      'kinetic energy: ',
      utils.calc_kinetic_energy(
        sample_fn, forward_fn, inverse_fn, params, rng, FLAGS.dim
      )
    )
    plt.savefig('results/fig/density_1d.pdf')
    plt.show()

    plot_1d_map = utils.plot_1d_map(
      forward_fn=forward_fn, params=params, final_mean=3
    )
    plot_1d_map

  elif FLAGS.dim == 2:
    fake_cond = np.zeros((FLAGS.batch_size, 1))
    samples = sample_fn(
      params, seed=key, sample_shape=(FLAGS.batch_size, ), cond=fake_cond
    )
    plt.scatter(samples[..., 0], samples[..., 1], s=1, c='r')
    fake_cond = np.ones((FLAGS.batch_size, 1))
    samples = sample_fn(
      params, seed=key, sample_shape=(FLAGS.batch_size, ), cond=fake_cond
    )
    plt.scatter(samples[..., 0], samples[..., 1], s=1, c='b')
    plt.savefig('results/fig/w1.pdf')
    plt.show()

    # plot the trajectory of the distribution and velocity field
    plot_traj_and_velocity = partial(
      utils.plot_traj_and_velocity,
      sample_fn=sample_fn,
      forward_fn=forward_fn,
      inverse_fn=inverse_fn,
      params=params,
      rng=rng
    )
    plot_traj_and_velocity(quiver_size=0.01)
    print(
      'kinetic energy: ',
      utils.calc_kinetic_energy(
        sample_fn, forward_fn, inverse_fn, params, rng, dim=FLAGS.dim
      )
    )

    print('kl loss: ', kl_loss_fn(params, rng, FLAGS.batch_size))


if __name__ == "__main__":
  app.run(main)
