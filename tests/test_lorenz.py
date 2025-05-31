import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

# Lorenz system parameters
sigma = 10.0
rho = 28.0
beta = 8/3

# Lorenz system equations
def lorenz_system(state, t):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return jnp.array([dxdt, dydt, dzdt])

# Euler-Maruyama method
def euler_maruyama_step(state, t, dt, noise):
    # Compute the deterministic part
    drift = lorenz_system(state, t)
    # Update state using Euler-Maruyama
    new_state = state + drift * dt + noise
    return new_state

# Simulation parameters
dt = 0.01  # Time step
T = 1.0   # Total time
N = int(T / dt)  # Number of steps

# Initial conditions
initial_state = jnp.array([1.0, 1.0, 1.0])
state = initial_state

# Random number generator
key = random.PRNGKey(0)

# Store results
results = jnp.zeros((N, 3))
results = results.at[0].set(state)

# Simulate the Lorenz system with Brownian motion
for i in range(1, N):
    t = i * dt
    key, subkey = random.split(key)
    # Generate Brownian motion increment
    noise = random.normal(subkey, shape=(3,)) * jnp.sqrt(dt) * 3
    state = euler_maruyama_step(state, t, dt, noise)
    results = results.at[i].set(state)

# Plot the results
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(results[:, 0], results[:, 1], results[:, 2])
ax.set_title('Lorenz System with Euler-Maruyama and Brownian Motion')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.savefig("lorenz.pdf")
plt.show()
              