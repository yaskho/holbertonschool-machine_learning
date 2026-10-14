#!/usr/bin/env python3
"""
Contains the expectation_maximization function for a GMM
"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs expectation maximization for a Gaussian Mixture Model

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        k: positive integer containing the number of clusters
        iterations: positive integer containing the maximum number of
                    iterations for the algorithm
        tol: non-negative float containing tolerance of the log likelihood
        verbose: boolean that determines if information should be printed

    Returns:
        pi, m, S, g, log_l or None, None, None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if type(k) is not int or k <= 0 or k > X.shape[0]:
        return None, None, None, None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, (float, int)) or tol < 0:
        return None, None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    g, log_l = expectation(X, pi, m, S)
    if g is None or log_l is None:
        return None, None, None, None, None

    if verbose:
        print(f"Log Likelihood after 0 iterations: {log_l:.5f}")

    for i in range(1, iterations + 1):
        l_prev = log_l

        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        g, log_l = expectation(X, pi, m, S)
        if g is None or log_l is None:
            return None, None, None, None, None

        if verbose and (i % 10 == 0 or abs(log_l - l_prev) <= tol or
                        i == iterations):
            print(f"Log Likelihood after {i} iterations: {log_l:.5f}")

        if abs(log_l - l_prev) <= tol:
            break

    return pi, m, S, g, log_l
