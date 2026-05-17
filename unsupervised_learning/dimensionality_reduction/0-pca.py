#!/usr/bin/env python3
"""PCA module."""

import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset.

    Args:
        X (np.ndarray): shape (n, d), centered data
        var (float): variance to preserve

    Returns:
        W (np.ndarray): weights matrix (d, nd)
    """

    # Step 1: SVD decomposition
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Step 2: compute cumulative variance ratio
    eigen_values = S ** 2
    total_variance = np.sum(eigen_values)
    variance_ratio = eigen_values / total_variance
    cumulative_variance = np.cumsum(variance_ratio)

    # Step 3: choose number of dimensions
    nd = np.argmax(cumulative_variance >= var) + 1

    # Step 4: construct W using first nd principal components
    W = Vt.T[:, :nd]

    return W
