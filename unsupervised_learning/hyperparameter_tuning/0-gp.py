#!/usr/bin/env python3
"""Gaussian Process module."""

import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian Process."""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Initialize the Gaussian Process.

        Args:
            X_init (numpy.ndarray): Sample inputs of shape (t, 1).
            Y_init (numpy.ndarray): Sample outputs of shape (t, 1).
            l (float): Length parameter for the RBF kernel.
            sigma_f (float): Signal standard deviation.
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        Compute the covariance kernel matrix using the RBF kernel.

        Args:
            X1 (numpy.ndarray): Array of shape (m, 1).
            X2 (numpy.ndarray): Array of shape (n, 1).

        Returns:
            numpy.ndarray: Covariance matrix of shape (m, n).
        """
        sqdist = (
            np.sum(X1 ** 2, axis=1, keepdims=True)
            + np.sum(X2 ** 2, axis=1)
            - 2 * np.matmul(X1, X2.T)
        )

        return (self.sigma_f ** 2) * np.exp(
            -sqdist / (2 * (self.l ** 2))
        )
