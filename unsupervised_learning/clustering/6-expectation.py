#!/usr/bin/env python3
"""
Expectation step for GMM (E-step)
"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Performs expectation step of EM algorithm.

    Returns:
        g: (k, n) responsibilities
        l: log likelihood
    """
    try:
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            return None, None
        if not isinstance(pi, np.ndarray) or not isinstance(m, np.ndarray):
            return None, None
        if not isinstance(S, np.ndarray):
            return None, None

        n, d = X.shape
        k = pi.shape[0]

        # compute P(k, n)
        P = np.zeros((k, n))

        # ONLY LOOP ALLOWED
        for i in range(k):
            P[i] = pdf(X, m[i], S[i]) * pi[i]

        # total probability per point
        norm = np.sum(P, axis=0)

        # responsibilities
        g = P / norm

        # log likelihood
        l = np.sum(np.log(norm))

        return g, l

    except Exception:
        return None, None
