#!/usr/bin/env python3
"""
Contains the function optimum_k that tests for the optimum number of clusters
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        kmin: positive integer containing the minimum number of clusters
              to check for (inclusive)
        kmax: positive integer containing the maximum number of clusters
              to check for (inclusive)
        iterations: positive integer containing the maximum number of
                    iterations for K-means

    Returns:
        results: list containing the outputs of K-means for each cluster size
        d_vars: list containing the difference in variance from the smallest
                cluster size for each cluster size
        or None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if type(kmax) is not int or kmax <= 0:
        return None, None
    if type(kmin) is not int or kmin < 1 or kmin >= kmax:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None

    results = []
    variances = []

    for k in range(kmin, kmax + 1):
        res = kmeans(X, k, iterations)
        if res is None or res[0] is None or res[1] is None:
            return None, None
        C, clss = res
        results.append(res)
        var = variance(X, C)
        if var is None:
            return None, None
        variances.append(var)

    if len(results) < 2:
        return None, None

    d_vars = [variances[0] - var for var in variances]

    return results, d_vars
