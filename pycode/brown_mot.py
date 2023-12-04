#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def brownian_motion(timesteps, dt):
    """Generate a 1D Brownian motion, this is the simple bachelier model with no drift."""
    t = np.linspace(0.0, 10.0, timesteps)
    W = np.cumsum(np.random.randn(timesteps))
    return t, W

def brownian_motion_var_and_drift(timesteps, dt, var, drift):
    """Generate a 1D Brownian with drift and instantaneous variance"""
    t = np.linspace(0.0, 10.0, timesteps)
    W = np.cumsum(drift*dt + var*np.random.randn(timesteps))
    return t, W

# Parameters
timesteps = 1000  # Number of time steps
dt = 0.01         # Time step size

# Generate three instances of Brownian motion
t1, W1 = brownian_motion_var_and_drift(timesteps, dt, 0, 100)
t2, W2 = brownian_motion_var_and_drift(timesteps, dt, 3, 100)
t3, W3 = brownian_motion_var_and_drift(timesteps, dt, 20, 100)

# Plot the Brownian motion
plt.figure(figsize=(10, 6))
plt.plot(t1, W1, label='1')
plt.plot(t2, W2, label='2')
plt.plot(t3, W3, label='3')
#plt.plot(t1, 0.2 * np.ones(timesteps), label="A")
# Customize the plot
plt.title('Arithmetic Brownian Motion with different parameters')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
