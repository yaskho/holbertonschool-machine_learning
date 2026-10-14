#!/usr/bin/env python3
"""
Variance calculation for K-means clustering
"""

import numpy as np


def variance(X, C):
    """
    Calculates total intra-cluster variance.

    Parameters:
        X (numpy.ndarray): data points (n, d)
        C (numpy.ndarray): centroids (k, d)

    Returns:
        float: total variance or None on failure
    """
    try:
        if not isinstance(X, np.ndarray) or not isinstance(C, np.ndarray):
            return None
        if len(X.shape) != 2 or len(C.shape) != 2:
            return None

        # compute distance from each point to each centroid
        distances = np.linalg.norm(X[:, None] - C, axis=2)  # (n, k)

        # assign each point to closest centroid
        closest = np.min(distances, axis=1)  # (n,)

        # total variance = sum of squared distances
        var = np.sum(closest ** 2)

        return var

    except Exception:
        return None
