#!/usr/bin/env python3
"""
Find optimal number of clusters using variance analysis
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests multiple k values and computes variance differences.

    Returns:
        results: list of (C, clss)
        d_vars: list of variance differences
    """
    try:
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            return None, None
        if not isinstance(kmin, int) or kmin <= 0:
            return None, None
        if kmax is None:
            kmax = X.shape[0]
        if not isinstance(kmax, int) or kmax < kmin:
            return None, None

        results = []
        variances = []

        # loop 1: run kmeans + compute variance
        for k in range(kmin, kmax + 1):
            C, clss = kmeans(X, k, iterations)
            results.append((C, clss))
            variances.append(variance(X, C))

        base_var = variances[0]

        # vectorized diff (no second loop over k range)
        d_vars = np.array(variances) - base_var

        return results, d_vars

    except Exception:
        return None, None
