#!/usr/bin/env python3
"""Bayesian Optimization module."""

import numpy as np
from scipy.stats import norm

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Bayesian Optimization using Gaussian Processes."""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Initialize Bayesian Optimization.

        Args:
            f: black-box function
            X_init: initial inputs (t, 1)
            Y_init: initial outputs (t, 1)
            bounds: (min, max)
            ac_samples: number of acquisition points
            l: kernel length
            sigma_f: signal variance
            xsi: exploration factor
            minimize: True for minimization, False for maximization
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)

        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)

        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Expected Improvement acquisition function.

        Returns:
            X_next: (1,)
            EI: (ac_samples,)
        """
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            best = np.min(self.gp.Y)
            imp = best - mu - self.xsi
        else:
            best = np.max(self.gp.Y)
            imp = mu - best - self.xsi

        sigma = np.maximum(sigma, 1e-9)

        Z = imp / sigma

        EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

        X_next = self.X_s[np.argmax(EI)].reshape(1)

        return X_next, EI

    def optimize(self, iterations=100):
        """
        Run Bayesian Optimization loop.

        Args:
            iterations: max number of steps

        Returns:
            X_opt: best input found (1,)
            Y_opt: best value found (1,)
        """
        for _ in range(iterations):
            X_next, _ = self.acquisition()

            # stop if already sampled
            if np.any(np.abs(self.gp.X - X_next) < 1e-8):
                break

            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)

        if self.minimize:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)

        return self.gp.X[idx].reshape(1), self.gp.Y[idx].reshape(1)
