#!/usr/bin/env python3
"""t-SNE gradients module."""

import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
    Calculates gradients of Y and Q affinities.

    Args:
        Y (np.ndarray): low-dimensional points (n, ndim)
        P (np.ndarray): high-dimensional affinities (n, n)

    Returns:
        dY (np.ndarray): gradients (n, ndim)
        Q (np.ndarray): low-dimensional affinities (n, n)
    """

    Q, _ = Q_affinities(Y)
    n, ndim = Y.shape

    dY = np.zeros((n, ndim))

    # symmetric term (P + P^T - Q - Q^T)
    PQ = P - Q

    for i in range(n):
        diff = Y[i] - Y
        coeff = PQ[i] + PQ[:, i]
        dY[i] = np.sum((coeff[:, None] * diff), axis=0)

    return dY, Q
