#!/usr/bin/env python3
"""PCA v2 module."""

import numpy as np


def pca(X, ndim):
    """
    Performs PCA and reduces X to ndim dimensions.

    Args:
        X (np.ndarray): shape (n, d)
        ndim (int): target dimensionality

    Returns:
        T (np.ndarray): shape (n, ndim)
    """

    # Step 1: center data
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    # Step 2: SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Step 3: projection matrix (principal components)
    W = Vt.T[:, :ndim]

    # Step 4: transform data
    T = np.matmul(X_centered, W)

    return T
