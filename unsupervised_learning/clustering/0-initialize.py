#!/usr/bin/env python3
import numpy as np

def initialize(X, k):
    """
    Initializes cluster centroids for K-means using uniform distribution.

    X: numpy.ndarray of shape (n, d)
    k: number of clusters

    Returns:
        numpy.ndarray of shape (k, d) or None on failure
    """
    try:
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            return None
        if not isinstance(k, int) or k <= 0:
            return None

        d = X.shape[1]

        # compute per-dimension min and max
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)

        # one call to uniform (required constraint)
        centroids = np.random.uniform(
            low=X_min,
            high=X_max,
            size=(k, d)
        )

        return centroids

    except Exception:
        return None
