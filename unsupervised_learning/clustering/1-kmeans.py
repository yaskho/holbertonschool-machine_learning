#!/usr/bin/env python3
"""
Contains the function kmeans that performs K-means clustering on a dataset
"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means using a multivariate uniform
    distribution.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer containing the number of clusters

    Returns:
        numpy.ndarray of shape (k, d) containing initialized centroids
    """
    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    return np.random.uniform(low=low, high=high, size=(k, X.shape[1]))


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
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    # Call initialize function (which uses np.random.uniform for the 1st time)
    C = initialize(X, k)
    if C is None:
        return None, None

    n, d = X.shape

    # Loop 1: Iterations loop
    for _ in range(iterations):
        C_prev = np.copy(C)

        # Vectorized Euclidean distance calculation: shape (n, k)
        distances = np.linalg.norm(X[:, np.newaxis, :] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        # Update step with 2nd Loop allowed for iteration over clusters
        # Loop 2: Cluster index loop
        for j in range(k):
            points = X[clss == j]
            if len(points) == 0:
                # Reinitialize centroid using 2nd call to np.random.uniform
                low = np.min(X, axis=0)
                high = np.max(X, axis=0)
                C[j] = np.random.uniform(low=low, high=high, size=(1, d))
            else:
                C[j] = np.mean(points, axis=0)

        # Recalculate classes after reinitialization check
        distances = np.linalg.norm(X[:, np.newaxis, :] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        # Stop if no change in centroids occurs
        if np.array_equal(C, C_prev):
            break

    return C, clss
