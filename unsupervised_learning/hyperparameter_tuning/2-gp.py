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
        Compute the RBF covariance kernel matrix.

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
            -sqdist / (2 * (self.l ** 2))
        )

    def predict(self, X_s):
        """
        Predict the mean and variance at sample points.

        Args:
            X_s: numpy.ndarray of shape (s, 1)

        Returns:
            mu: numpy.ndarray of shape (s,)
            sigma: numpy.ndarray of shape (s,)
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu = (K_s.T @ K_inv @ self.Y).reshape(-1)
        sigma = np.diag(K_ss - K_s.T @ K_inv @ K_s)

        return mu, sigma

    def update(self, X_new, Y_new):
        """
        Update the Gaussian Process with a new sample.

        Args:
            X_new: numpy.ndarray of shape (1,)
            Y_new: numpy.ndarray of shape (1,)
        """
        self.X = np.vstack((self.X, X_new.reshape(1, 1)))
        self.Y = np.vstack((self.Y, Y_new.reshape(1, 1)))
        self.K = self.kernel(self.X, self.X)
