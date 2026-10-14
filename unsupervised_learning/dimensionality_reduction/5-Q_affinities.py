#!/usr/bin/env python3
"""t-SNE Q affinities module."""

import numpy as np


def Q_affinities(Y):
    """
    Calculates Q affinities for t-SNE.

    Args:
        Y (np.ndarray): shape (n, ndim), low-dimensional embedding

    Returns:
        Q (np.ndarray): normalized affinities (n, n)
        num (np.ndarray): numerator of Q affinities
    """

    n = Y.shape[0]

    # pairwise squared distances in low-dimensional space
    sum_Y = np.sum(np.square(Y), axis=1)
    D = sum_Y.reshape((n, 1)) + sum_Y.reshape((1, n)) - 2 * np.dot(Y, Y.T)

    # Student-t kernel numerator
    num = 1 / (1 + D)

    # remove diagonal (no self-similarity)
    np.fill_diagonal(num, 0)

    # normalization term
    Z = np.sum(num)

    Q = num / Z

    return Q, num
