#!/usr/bin/env python3
"""Module to calculate Bayesian Information Criterion (BIC) for GMM."""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using BIC.

    Parameters:
        X: numpy.ndarray of shape (n, d) containing the data set
        kmin: positive integer, minimum number of clusters to check
        kmax: positive integer, maximum number of clusters to check
        iterations: positive integer, maximum number of iterations for EM
        tol: non-negative float, tolerance for EM
        verbose: boolean, determines if EM prints information

    Returns:
        best_k: best value for k based on BIC
        best_result: tuple containing (pi, m, S) for best_k
        l: numpy.ndarray of shape (kmax - kmin + 1) with log likelihoods
        b: numpy.ndarray of shape (kmax - kmin + 1) with BIC values
        or (None, None, None, None) on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None

    n, d = X.shape

    if type(kmin) is not int or kmin <= 0 or kmin > n:
        return None, None, None, None

    if kmax is None:
        kmax = n

    if type(kmax) is not int or kmax <= 0 or kmax > n:
        return None, None, None, None

    if kmin > kmax:
        return None, None, None, None

    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None

    if type(tol) is not float or tol < 0:
        return None, None, None, None

    if type(verbose) is not bool:
        return None, None, None, None

    l_all = []
    b_all = []
    results = []

    for k in range(kmin, kmax + 1):
        pi, m, S, _, l_k = expectation_maximization(
            X, k, iterations, tol, verbose
        )
        if l_k is None:
            return None, None, None, None

        l_all.append(l_k)
        results.append((pi, m, S))

        p = k - 1 + k * d + k * d * (d + 1) // 2
        bic = p * np.log(n) - 2 * l_k
        b_all.append(bic)

    l_all = np.array(l_all)
    b_all = np.array(b_all)

    best_idx = np.argmin(b_all)
    best_k = kmin + best_idx
    best_result = results[best_idx]

    return best_k, best_result, l_all, b_all
