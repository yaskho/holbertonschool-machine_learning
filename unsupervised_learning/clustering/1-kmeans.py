#!/usr/bin/env python3
"""
Contains the initialize and kmeans functions for K-means clustering
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
        numpy.ndarray of shape (k, d) containing initialized centroids,
        or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if type(k) is not int or k <= 0 or k > X.shape[0]:
        return None

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    return np.random.uniform(low, high, size=(k, X.shape[1]))


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
    if type(k) is not int or k <= 0 or k > X.shape[0]:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None

    C = initialize(X, k)
    if C is None:
        return None, None

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)

    for i in range(iterations):
        C_prev = np.copy(C)

        distances = np.linalg.norm(X[:, np.newaxis, :] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        for j in range(k):
            points = X[clss == j]
            if len(points) == 0:
                C[j] = np.random.uniform(low, high)
            else:
                C[j] = np.mean(points, axis=0)

        if np.array_equal(C, C_prev):
            break

    distances = np.linalg.norm(X[:, np.newaxis, :] - C, axis=2)
    clss = np.argmin(distances, axis=1)

    return C, clss
