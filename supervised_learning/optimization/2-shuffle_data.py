#!/usr/bin/env python3
"""
Shuffles two matrices (X and Y) in the same way.
"""
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way.

    Parameters
    ----------
    X : numpy.ndarray of shape (m, nx)
        First dataset to shuffle.
    Y : numpy.ndarray of shape (m, ny)
        Second dataset to shuffle.

    Returns
    -------
    X_shuffled : numpy.ndarray
        Shuffled X.
    Y_shuffled : numpy.ndarray
        Shuffled Y.
    """
    m = X.shape[0]
    permutation = np.random.permutation(m)

    return X[permutation], Y[permutation]
