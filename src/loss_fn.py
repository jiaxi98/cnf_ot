from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from src.types import PRNGKey

# loss function for training, including the KL-divergence at the boundary condition 
  # and kinetic energy along the trajectory
@partial(jax.jit, static_argnames=['batch_size'])
def kl_loss_fn(params: hk.Params, rng: PRNGKey, cond, target_prob_fn, batch_size: int) -> Array:
    """KL-divergence between the normalizing flow and the target distribution.

    TODO: here, we assume the p.d.f. of the target distribution is known. 
    In the case where we only access to samples from target distribution,
    KL-divergence is not calculable and we need to shift to other integral 
    probability metric, e.g. MMD.
    """

    fake_cond_ = np.ones((batch_size, 1)) * cond
    samples, log_prob = model.apply.sample_and_log_prob(
        params,
        cond=fake_cond_,
        seed=rng,
        sample_shape=(batch_size, ),
    )
    return (log_prob - jnp.log(target_prob_fn(samples))).mean()

@partial(jax.jit, static_argnames=['batch_size'])
def double_kl_loss_fn(params: hk.Params, rng: PRNGKey, batch_size: int) -> Array:

    return kl_loss_fn(hk.Params, PRNGKey, 0, source_prob, batch_size) + \
        kl_loss_fn(hk.Params, PRNGKey, 1, target_prob, batch_size)

@partial(jax.jit, static_argnames=['batch_size'])
def kinetic_loss_fn(t: float, params: hk.Params, rng: PRNGKey, batch_size: int) -> Array:
    """Kinetic energy along the trajectory at time t
    """
    fake_cond_ = np.ones((batch_size, 1)) * t
    samples = sample_fn(params, seed=rng, sample_shape=(batch_size, ), cond=fake_cond_)
    xi = inverse_fn(params, samples, fake_cond_)
    velocity = jax.jacfwd(partial(forward_fn, params, xi))(fake_cond_)
    # velocity.shape = [batch_size, 2, batch_size, 1]
    # velocity[jnp.arange(batch_size),:,jnp.arange(batch_size),0].shape = [batch_size, 2]
    weight = .01
    return jnp.mean(velocity[jnp.arange(batch_size),:,jnp.arange(batch_size),0]**2) * weight * FLAGS.dim / 2

@partial(jax.jit, static_argnames=['batch_size'])
def acc_loss_fn(t: float, params: hk.Params, rng: PRNGKey, batch_size: int) -> Array:
    """acceleration energy along the trajectory at time t, used for regularization
    """
    fake_cond_ = np.ones((batch_size, 1)) * t
    samples = sample_fn(params, seed=rng, sample_shape=(batch_size, ), cond=fake_cond_)
    xi = inverse_fn(params, samples, fake_cond_)
    velocity = jax.jacfwd(partial(forward_fn, params, xi))(fake_cond_)
    # velocity.shape = [batch_size, 2, batch_size, 1]
    # velocity[jnp.arange(batch_size),:,jnp.arange(batch_size),0].shape = [batch_size, 2]

    dt = 0.01
    fake_cond_ = np.ones((batch_size, 1)) * (t+dt)
    samples = sample_fn(params, seed=rng, sample_shape=(batch_size, ), cond=fake_cond_)
    xi = inverse_fn(params, samples, fake_cond_)
    velocity_ = jax.jacfwd(partial(forward_fn, params, xi))(fake_cond_)
    #weight = .01
    return jnp.mean(velocity[jnp.arange(batch_size),:,jnp.arange(batch_size),0] - velocity_[jnp.arange(batch_size),:,jnp.arange(batch_size),0]) * FLAGS.dim / 2

@partial(jax.jit, static_argnames=['batch_size'])
def potential_loss_fn(params: hk.Params, rng: PRNGKey, cond, potential_fn, batch_size: int) -> Array:

    fake_cond_ = np.ones((batch_size, 1)) * cond
    samples, _ = model.apply.sample_and_log_prob(
        params,
        cond=fake_cond_,
        seed=rng,
        sample_shape=(batch_size, ),
    )
    return potential_fn(samples).mean()

@partial(jax.jit, static_argnames=['batch_size'])
def w_loss_fn(params: hk.Params, rng: PRNGKey, batch_size: int) -> Array:
    """Loss of the Wasserstein gradient flow, including:

    KL divergence between the normalizing flow at t=0 and the source distribution
    KL divergence between the normalizing flow at t=1 and the target distribution
    Monte-Carlo integration of the kinetic energy along the interval [0, 1]

    TODO: one caveat of the coding here is that we do not further split the 
    rng for sampling from CNF in kl_loss and kinetic_loss. 
    """

    loss = double_kl_loss_fn(params, rng, batch_size)
    # t_batch_size = 10 # 10
    # t_batch = jax.random.uniform(rng, (t_batch_size, ))
    # for _ in range(t_batch_size):
    #   loss += kinetic_loss_fn(t_batch[_], params, rng, batch_size//32)/t_batch_size + acc_loss_fn(t_batch[_], params, rng, batch_size//32)/t_batch_size

    return loss

@partial(jax.jit, static_argnames=['batch_size'])
def mfg_loss_fn(params: hk.Params, rng: PRNGKey, batch_size: int) -> Array:

    return kl_loss_fn(hk.Params, PRNGKey, 0, source_prob, batch_size) + \
        potential_loss_fn(hk.Params, PRNGKey, 1, potential_fn, batch_size)