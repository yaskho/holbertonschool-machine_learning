#!/usr/bin/env python3
"""
Gaussian Process initialization module
"""
import numpy as np


class GaussianProcess:
    """
    Represents a noiseless 1D Gaussian process
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Class constructor for GaussianProcess

        Parameters:
            X_init: numpy.ndarray of shape (t, 1) - sampled inputs
            Y_init: numpy.ndarray of shape (t, 1) - sampled outputs
            l: length scale parameter for the kernel
            sigma_f: standard deviation multiplier for kernel output
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix between two matrices
        using the Radial Basis Function (RBF) kernel

        Parameters:
            X1: numpy.ndarray of shape (m, 1)
            X2: numpy.ndarray of shape (n, 1)

        Returns:
            Covariance kernel matrix as a numpy.ndarray of shape (m, n)
        """
        sqdist = (X1 - X2.T) ** 2
        return (self.sigma_f ** 2) * np.exp(-0.5 * sqdist / (self.l ** 2))
