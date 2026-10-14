#!/usr/bin/env python3
"""t-SNE initialization module."""

import numpy as np


def P_init(X, perplexity):
    """
    Initializes variables for t-SNE P affinities.

    Args:
        X (np.ndarray): dataset (n, d)
        perplexity (float): perplexity value

    Returns:
        D (np.ndarray): pairwise squared distances
        P (np.ndarray): zeros matrix
        betas (np.ndarray): beta values (ones)
        H (float): entropy (log2 perplexity)
    """

    n = X.shape[0]

    # pairwise squared distances
    sum_X = np.sum(X ** 2, axis=1)
    D = sum_X.reshape((n, 1)) + sum_X.reshape((1, n)) - 2 * np.dot(X, X.T)

    np.fill_diagonal(D, 0)

    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)

    return D, P, betas, H
