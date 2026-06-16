#!/usr/bin/env python3
"""Bayesian Optimization module."""

import numpy as np
from scipy.stats import norm

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Bayesian Optimization using Gaussian Processes."""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """Initialize Bayesian Optimization."""
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)

        self.X_s = np.linspace(
            bounds[0], bounds[1], ac_samples
        ).reshape(-1, 1)

        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculate next best sample location using Expected Improvement.

        Returns:
            X_next (numpy.ndarray): shape (1,)
            EI (numpy.ndarray): shape (ac_samples,)
        """
        mu, sigma = self.gp.predict(self.X_s)

        mu_sample_opt = np.min(self.gp.Y) if self.minimize else np.max(self.gp.Y)

        if self.minimize:
            imp = mu_sample_opt - mu - self.xsi
        else:
            imp = mu - mu_sample_opt - self.xsi

        sigma = np.maximum(sigma, 1e-9)  # avoid division by zero

        Z = imp / sigma

        EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

        X_next = self.X_s[np.argmax(EI)].reshape(1)

        return X_next, EI
