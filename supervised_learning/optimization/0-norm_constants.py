#!/usr/bin/env python3
import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization constants of a matrix.

    Parameters
    ----------
    X : numpy.ndarray of shape (m, nx)
        Matrix containing the data set.

    Returns
    -------
    mean : numpy.ndarray of shape (nx,)
        Mean of each feature.
    std : numpy.ndarray of shape (nx,)
        Standard deviation of each feature.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return mean, std
