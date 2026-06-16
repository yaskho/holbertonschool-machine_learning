#!/usr/bin/env python3
"""Bayesian Optimization module."""

import numpy as np

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D Gaussian process."""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Initialize the Bayesian optimization object.

        Args:
            f: Black-box function to optimize.
            X_init: numpy.ndarray of shape (t, 1) containing sampled inputs.
            Y_init: numpy.ndarray of shape (t, 1) containing sampled outputs.
            bounds: Tuple (min, max) defining the search interval.
            ac_samples: Number of acquisition sample points.
            l: Length parameter for the RBF kernel.
            sigma_f: Signal standard deviation.
            xsi: Exploration-exploitation factor.
            minimize: True for minimization, False for maximization.
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(
            bounds[0],
            bounds[1],
            ac_samples
        ).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize
