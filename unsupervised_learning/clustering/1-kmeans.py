#!/usr/bin/env python3
"""
Contains the function kmeans that performs K-means clustering on a dataset
"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer containing the number of clusters
        iterations: positive integer containing the maximum number of
                    iterations to perform

    Returns:
        tuple: (C, clss) or (None, None) on failure
               C: numpy.ndarray of shape (k, d) containing centroid means
               clss: numpy.ndarray of shape (n,) containing the index of the
                     cluster in C that each data point belongs to
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or X.shape[0] < k:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)

    # 1st call to np.random.uniform
    C = np.random.uniform(low=low, high=high, size=(k, X.shape[1]))

    # Loop 1: Iterations loop
    for _ in range(iterations):
        C_prev = np.copy(C)

        # Distance calculation
        distances = np.linalg.norm(X[:, np.newaxis, :] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        # 2nd call to np.random.uniform (covers potential empty clusters)
        reinit = np.random.uniform(low=low, high=high, size=(k, X.shape[1]))

        # Loop 2: Cluster loop
        for j in range(k):
            points = X[clss == j]
            if len(points) == 0:
                C[j] = reinit[j]
            else:
                C[j] = np.mean(points, axis=0)

        # Recalculate classes after centroid update
        distances = np.linalg.norm(X[:, np.newaxis, :] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        if np.array_equal(C, C_prev):
            break

    return C, clss
