#!/usr/bin/env python3
"""Module that defines a Multivariate Normal distribution"""
import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution"""

    def __init__(self, data):
        """
        Class constructor

        data is a numpy.ndarray of shape (d, n) containing the data set:
            n is the number of data points
            d is the number of dimensions in each data point

        Sets the public instance variables:
            mean - numpy.ndarray of shape (d, 1) with the mean of data
            cov - numpy.ndarray of shape (d, d) with the covariance matrix
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        data_centered = data - self.mean
        self.cov = np.matmul(data_centered, data_centered.T) / (n - 1)

    def pdf(self, x):
        """
        Calculates the PDF at a data point

        x is a numpy.ndarray of shape (d, 1) containing the data point
            whose PDF should be calculated
            d is the number of dimensions of the MultiNormal instance

        Returns: the value of the PDF
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.cov.shape[0]
        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        diff = x - self.mean
        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)

        norm = 1 / np.sqrt(((2 * np.pi) ** d) * det)
        exponent = -0.5 * np.matmul(np.matmul(diff.T, inv), diff)

        return float(norm * np.exp(exponent[0, 0]))
