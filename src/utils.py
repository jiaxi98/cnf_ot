from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
from matplotlib import pyplot as plt

from src.types import Batch, OptState, PRNGKey

def calculate_kinetic_energy(
        sample_fn, 
        forward_fn, 
        inverse_fn, 
        params: hk.Params, 
        rng: PRNGKey,
        dim: int=1):

    t_array = jnp.linspace(0, 1, 100)
    batch_size = 1024
    kinetic_energy = 0
    for t in t_array:

        key, rng = jax.random.split(rng)
        fake_cond_ = np.ones((batch_size, 1)) * t
        samples = sample_fn(params, seed=key, sample_shape=(batch_size, ), cond=fake_cond_)
        xi = inverse_fn(params, samples, fake_cond_)
        velocity = jax.jacfwd(partial(forward_fn, params, xi))(fake_cond_)
        kinetic_energy += jnp.mean(velocity[jnp.arange(batch_size),:,jnp.arange(batch_size),0]**2)
    
    return kinetic_energy/100 * dim 

def plot_distribution_at_time(params: hk.Params, rng: PRNGKey):
    return

def plot_traj_and_velocity(
    sample_fn, 
    forward_fn, 
    inverse_fn, 
    params: hk.Params, 
    rng: PRNGKey,
    quiver_size: float=.1):

    t_array = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
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
        samples = sample_fn(params, seed=key, sample_shape=(batch_size_pdf, ), cond=fake_cond_)
        ax1[i//2, i%2].scatter(samples[...,0], samples[...,1], s=1)

        fake_cond_ = np.ones((batch_size_velocity, 1)) * t
        samples = sample_fn(params, seed=key, sample_shape=(batch_size_velocity, ), cond=fake_cond_)
        xi = inverse_fn(params, samples, fake_cond_)
        velocity = jax.jacfwd(partial(forward_fn, params, xi))(fake_cond_)
        #breakpoint()
        ax2[i//2, i%2].quiver(
            samples[...,0], 
            samples[...,1], 
            velocity[jnp.arange(batch_size_velocity),0,jnp.arange(batch_size_velocity),0], 
            velocity[jnp.arange(batch_size_velocity),1,jnp.arange(batch_size_velocity),0],) 
            #scale=quiver_size)
        i += 1
    plt.savefig('results/fig/traj.pdf')
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
        x_axis = np.linspace(-3,3,batch_size_pdf).reshape(-1,1)
        y_axis = forward_fn(params,x_axis,fake_cond_)
        true_y = x_axis + final_mean*t 
        ax1[i//2, i%2].plot(x_axis,y_axis,'b')
        ax1[i//2, i%2].plot(x_axis,true_y,'r')
        i += 1
    plt.savefig('results/fig/mapping_1d.pdf')
    # plt.clf()
    plt.show()