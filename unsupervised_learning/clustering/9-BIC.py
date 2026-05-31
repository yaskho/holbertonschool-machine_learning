#!/usr/bin/env python3
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n

    if kmin < 1 or kmin > kmax:
        return None, None, None, None

    ks = list(range(kmin, kmax + 1))

    l_vals = []
    b_vals = []
    results = []

    for k in ks:
        pi, m, S, g, l = expectation_maximization(X, k, iterations, tol, verbose)

        if pi is None:
            return None, None, None, None

        # number of parameters in GMM
        p = (k * d) + (k * d * (d + 1)) // 2 + (k - 1)

        bic = p * np.log(n) - 2 * l

        l_vals.append(l)
        b_vals.append(bic)
        results.append((pi, m, S))

    l_vals = np.array(l_vals)
    b_vals = np.array(b_vals)

    best_idx = np.argmin(b_vals)

    best_k = ks[best_idx]
    best_result = results[best_idx]

    return best_k, best_result, l_vals, b_vals
