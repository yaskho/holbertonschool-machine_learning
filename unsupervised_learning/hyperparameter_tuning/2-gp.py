Here is the implementation for 2-gp.py with the update method added:

Python
#!/usr/bin/env python3
"""
Gaussian Process update module
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

    def predict(self, X_s):
        """
        Predicts the mean and variance of points in a Gaussian process

        Parameters:
            X_s: numpy.ndarray of shape (s, 1) containing all points
                 whose mean and variance should be calculated

        Returns:
            mu: numpy.ndarray of shape (s,) containing the mean for
                each point in X_s
            sigma: numpy.ndarray of shape (s,) containing the variance for
                   each point in X_s
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu = K_s.T @ K_inv @ self.Y
        mu = mu.reshape(-1)

        sigma_cov = K_ss - K_s.T @ K_inv @ K_s
        sigma = np.diag(sigma_cov)

        return mu, sigma

    def update(self, X_new, Y_new):
        """
        Updates a Gaussian Process with a new sample point and value

        Parameters:
            X_new: numpy.ndarray of shape (1,) - new sample point
            Y_new: numpy.ndarray of shape (1,) - new sample function value
        """
        self.X = np.vstack((self.X, X_new.reshape(-1, 1)))
        self.Y = np.vstack((self.Y, Y_new.reshape(-1, 1)))
        self.K = self.kernel(self.X, self.X)
