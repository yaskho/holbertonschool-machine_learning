#!/usr/bin/env python3
"""
Expectation-Maximization for Gaussian Mixture Model
"""

import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs EM algorithm for GMM.

    Returns:
        pi, m, S, g, l
    """
    try:
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            return None, None, None, None, None
        if not isinstance(k, int) or k <= 0:
            return None, None, None, None, None

        pi, m, S = initialize(X, k)

        prev_l = None

        for i in range(iterations + 1):  # single loop

            g, l = expectation(X, pi, m, S)

            if verbose and (i % 10 == 0 or i == iterations):
                print(f"Log Likelihood after {i} iterations: {l:.5f}")

            if prev_l is not None and abs(l - prev_l) <= tol:
                break

            prev_l = l

            if i == iterations:
                break

            pi, m, S = maximization(X, g)

        return pi, m, S, g, l

    except Exception:
        return None, None, None, None, None
