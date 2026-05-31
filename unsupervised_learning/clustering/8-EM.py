#!/usr/bin/env python3
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None:
        return None, None, None, None, None

    prev_l = None

    for i in range(iterations + 1):
        g, l = expectation(X, pi, m, S)

        if verbose and (i % 10 == 0 or i == 0):
            print(f"Log Likelihood after {i} iterations: {l:.5f}")

        # convergence check AFTER logging logic
        if prev_l is not None and abs(l - prev_l) <= tol:
            if verbose and i % 10 != 0:
                print(f"Log Likelihood after {i} iterations: {l:.5f}")
            break

        prev_l = l
        pi, m, S = maximization(X, g)

    return pi, m, S, g, l
