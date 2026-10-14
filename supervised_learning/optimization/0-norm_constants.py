#!/usr/bin/env python3
"""
Module to calculate normalization constants of a matrix.
"""

import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization (standardization) constants of a matrix.

    Args:
        X (numpy.ndarray): Matrix of shape (m, nx) to normalize.
            m is the number of data points.
            nx is the number of features.

    Returns:
        tuple: (mean, std) containing the mean and standard deviation
            of each feature, respectively.
    """
    return np.mean(X, axis=0), np.std(X, axis=0)
