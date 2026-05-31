#!/usr/bin/env python3
"""
Maximization step for GMM (M-step)
"""

import numpy as np


def maximization(X, g):
    """
    Performs maximization step of EM algorithm.

    Parameters:
        X (n, d): dataset
        g (k, n): responsibilities

    Returns:
        pi (k,), m (k, d), S (k, d, d)
    """
    try:
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            return None, None, None
        if not isinstance(g, np.ndarray) or len(g.shape) != 2:
            return None, None, None

        n, d = X.shape
        k = g.shape[0]

        Nk = np.sum(g, axis=1)  # (k,)

        # priors
        pi = Nk / n

        m = np.zeros((k, d))
        S = np.zeros((k, d, d))

        for i in range(k):  # allowed single loop
            # mean
            m[i] = np.sum(g[i][:, None] * X, axis=0) / Nk[i]

            # covariance
            diff = X - m[i]
            S[i] = (g[i][:, None] * diff).T @ diff / Nk[i]

        return pi, m, S

    except Exception:
        return None, None, None
