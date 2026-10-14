#!/usr/bin/env python3
"""
Contains the BIC function for a Gaussian Mixture Model
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using BIC

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        kmin: positive integer containing the minimum number of clusters
        kmax: positive integer containing the maximum number of clusters
        iterations: positive integer containing max iterations for EM
        tol: non-negative float containing tolerance for EM
        verbose: boolean that determines if EM prints info

    Returns:
        best_k, best_result, l, b or None, None, None, None on failure
        best_k: best value for k based on BIC
        best_result: tuple containing (pi, m, S) for best k
        l: numpy.ndarray of shape (kmax - kmin + 1) with log likelihoods
        b: numpy.ndarray of shape (kmax - kmin + 1) with BIC values
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, (float, int)) or tol < 0:
        return None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n

    if type(kmin) is not int or kmin <= 0 or kmin > kmax:
        return None, None, None, None
    if type(kmax) is not int or kmax <= 0 or kmax > n:
        return None, None, None, None

    l_list = []
    b_list = []
    results = []

    cov_params = d * (d + 1) / 2

    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_l = expectation_maximization(
            X, k, iterations, tol, verbose
        )
        if pi is None or m is None or S is None or g is None or log_l is None:
            return None, None, None, None

        results.append((pi, m, S))
        l_list.append(log_l)

        p = (k - 1) + (k * d) + (k * cov_params)
        bic = p * np.log(n) - 2 * log_l
        b_list.append(bic)

    l_arr = np.array(l_list)
    b_arr = np.array(b_list)

    best_idx = np.argmin(b_arr)
    best_k = kmin + best_idx
    best_result = results[best_idx]

    return best_k, best_result, l_arr, b_arr
