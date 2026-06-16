#!/usr/bin/env python3
"""Gaussian Process module."""

import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian Process."""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Initialize the Gaussian Process.

        Args:
            X_init: numpy.ndarray of shape (t, 1)
            Y_init: numpy.ndarray of shape (t, 1)
            l: length parameter of the RBF kernel
            sigma_f: standard deviation of the output
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        Calculate the covariance kernel matrix using the RBF kernel.

        Args:
            X1: numpy.ndarray of shape (m, 1)
            X2: numpy.ndarray of shape (n, 1)

        Returns:
            numpy.ndarray of shape (m, n)
        """
        sqdist = (
            np.sum(X1 ** 2, axis=1, keepdims=True)
            + np.sum(X2 ** 2, axis=1)
            - 2 * np.matmul(X1, X2.T)
        )

        return (self.sigma_f ** 2) * np.exp(
            -sqdist / (2 * self.l ** 2)
        )

    def predict(self, X_s):
        """
        Predict the mean and variance of points in the Gaussian Process.

        Args:
            X_s: numpy.ndarray of shape (s, 1)

        Returns:
            mu: numpy.ndarray of shape (s,)
            sigma: numpy.ndarray of shape (s,)
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu = K_s.T @ K_inv @ self.Y
        mu = mu.reshape(-1)

        cov = K_ss - K_s.T @ K_inv @ K_s
        sigma = np.diag(cov)

        return mu, sigma
