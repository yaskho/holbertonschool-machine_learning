#!/usr/bin/env python3
"""
K-means clustering module
"""

import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids using uniform distribution.
    """
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)

    return np.random.uniform(low=X_min, high=X_max, size=(k, X.shape[1]))


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering.

    Returns:
        C: centroids (k, d)
        clss: cluster assignments (n,)
    """
    try:
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            return None, None
        if not isinstance(k, int) or k <= 0:
            return None, None
        if not isinstance(iterations, int) or iterations <= 0:
            return None, None

        n, d = X.shape

        # 1st call to uniform (via initialize)
        C = initialize(X, k)

        for _ in range(iterations):
            # compute distances (vectorized)
            distances = np.linalg.norm(X[:, None] - C, axis=2)
            clss = np.argmin(distances, axis=1)

            new_C = np.zeros((k, d))

            # update centroids
            for i in range(k):  # allowed loop #1
                points = X[clss == i]

                if len(points) == 0:
                    # 2nd and last uniform call (reinitialization)
                    new_C[i] = np.random.uniform(
                        low=X.min(axis=0),
                        high=X.max(axis=0),
                        size=(d,)
                    )
                else:
                    new_C[i] = points.mean(axis=0)

            # convergence check
            if np.all(C == new_C):
                return C, clss

            C = new_C

        return C, clss

    except Exception:
        return None, None
