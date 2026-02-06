"""
Docstring for FinTDA.henon_map

    Generating stochastic time series using the Henon map, a discrete-time dynamical system that exhibits chaotic behavior. The Henon map is defined by the equations:
    x_{n+1} = 1 - a_n * x_n^2 + b * y_n + sigma * W_n sqrt(delta_t)
    y_{n+1} = x_n+1 + sigma * W_n sqrt(delta_t)
    a_n+1 = a_n + sqrt(delta_t)

    where
    - W_n is a standard normal random variable (Gaussian noise),
    - a_n is a time-varying parameter that evolves linearly with time,
    - b is a constant parameter,
    - sigma controls the noise intensity (positive value),
    - delta_t is the time step size (positive value).
"""

import numpy as np


class HenonMap:
    def __init__(self, a_0=0.0, b=0.3, sigma=0.1, delta_t=0.01, seed=None):
        self.a_0 = a_0
        self.b = b
        self.sigma = sigma
        self.delta_t = delta_t
        self._rng = np.random.default_rng(seed)
        self._cache = None

    def reset_cache(self):
        self._cache = None

    def generate_time_series(self, initial_conditions=(0.0, 0.0), n_steps=100, use_cache=True):
        if n_steps <= 0:
            return np.array([]), np.array([]), np.array([])

        if not use_cache or self._cache is None:
            x = np.zeros(n_steps)
            y = np.zeros(n_steps)
            a_values = np.zeros(n_steps)

            # Initial conditions
            x[0] = initial_conditions[0]
            y[0] = initial_conditions[1]
            a_values[0] = self.a_0

            start_idx = 1
        else:
            x, y, a_values, cached_steps = self._cache
            if n_steps <= cached_steps:
                return x[:n_steps].copy(), y[:n_steps].copy(), a_values[:n_steps].copy()

            x = np.resize(x, n_steps)
            y = np.resize(y, n_steps)
            a_values = np.resize(a_values, n_steps)
            start_idx = cached_steps

        for n in range(start_idx, n_steps):
            W_n = self._rng.normal(0, 1)  # Standard normal random variable
            x[n] = 1 - a_values[n-1] * x[n-1]**2 + self.b * y[n-1] + self.sigma * W_n * np.sqrt(self.delta_t)
            y[n] = x[n] + self.sigma * W_n * np.sqrt(self.delta_t)
            a_values[n] = a_values[n-1] + np.sqrt(self.delta_t)

        if use_cache:
            self._cache = (x, y, a_values, n_steps)

        return x, y, a_values
