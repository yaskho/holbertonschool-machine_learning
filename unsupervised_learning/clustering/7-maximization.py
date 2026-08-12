#!/usr/bin/env python3
"""
Contains the maximization function for a Gaussian Mixture Model
"""
import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        g: numpy.ndarray of shape (k, n) containing posterior probabilities

    Returns:
        pi, m, S, or None, None, None on failure
        pi: numpy.ndarray of shape (k,) containing updated priors
        m: numpy.ndarray of shape (k, d) containing updated centroid means
        S: numpy.ndarray of shape (k, d, d) containing updated covariances
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    n, d = X.shape
    k, n_g = g.shape

    if n != n_g:
        return None, None, None
    if not np.isclose(np.sum(g, axis=0), 1).all():
        return None, None, None

    Nk = np.sum(g, axis=1)
    pi = Nk / n

    m = np.matmul(g, X) / Nk[:, None]

    S = np.zeros((k, d, d))
    for i in range(k):
        diff = X - m[i]
        S[i] = np.matmul(g[i] * diff.T, diff) / Nk[i]

    return pi, m, S
