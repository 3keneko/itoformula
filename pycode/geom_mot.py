#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def geometric_brownian_motion(T, mu, sigma, S0, n, M):
    """
    Generate geometric Brownian motion paths.

    Parameters:
        T (float): Total time in years.
        mu (float): Drift (average return).
        sigma (float): Volatility (standard deviation of return).
        S0 (float): Initial value of the process.
        dt (float): Time step size.

    Returns:
        numpy.ndarray: Array of GBM values over time.
    """
    dt = T / n
    St = np.exp(
        (mu - 0.5 * sigma**2) * dt
        + sigma * np.random.normal(0, np.sqrt(dt), size=(M, n)).T
    )
    St = np.vstack([np.ones(M), St])
    St = S0 * St.cumprod(axis=0)

    return St

# Parameters
T = 2.0  # Total time in years
mu = 0.1  # Drift
sigma = 0.3  # Volatility
S0 = 100.0  # Initial value
steps = 100 # number of steps
M = 100 # number of simulations
# Generate GBM path
St = geometric_brownian_motion(T, mu, sigma, S0, steps, M)
time = np.linspace(0, T, steps+1)

tt = np.full(shape=(M, steps+1), fill_value=time).T
# Plot the results
plt.plot(tt, St)
plt.title('Geometric Brownian Motion')
plt.xlabel('Time (Years)')
plt.ylabel('Asset Price')
plt.show()
