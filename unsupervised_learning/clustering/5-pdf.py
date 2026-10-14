#!/usr/bin/env python3
"""
Multivariate Gaussian PDF
"""

import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution.

    Parameters:
        X (n, d): data points
        m (d,): mean vector
        S (d, d): covariance matrix

    Returns:
        (n,) array of probabilities or None
    """
    try:
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            return None
        if not isinstance(m, np.ndarray) or not isinstance(S, np.ndarray):
            return None

        n, d = X.shape

        diff = X - m

        S_inv = np.linalg.inv(S)
        det_S = np.linalg.det(S)

        norm_const = 1 / (((2 * np.pi) ** (d / 2)) * np.sqrt(det_S))

        exp_term = np.sum((diff @ S_inv) * diff, axis=1)

        P = norm_const * np.exp(-0.5 * exp_term)

        return np.maximum(P, 1e-300)

    except Exception:
        return None
