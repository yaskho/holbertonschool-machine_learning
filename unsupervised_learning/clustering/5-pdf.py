#!/usr/bin/env python3
"""
Multivariate Gaussian PDF
"""

import numpy as np


def pdf(X, m, S):
    """
    Calculates probability density function of a Gaussian distribution.

    Parameters:
        X (n, d): data points
        m (d,): mean vector
        S (d, d): covariance matrix

    Returns:
        P (n,): PDF values
    """
    try:
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            return None
        if not isinstance(m, np.ndarray) or not isinstance(S, np.ndarray):
            return None

        n, d = X.shape

        # reshape mean for broadcasting
        diff = X - m

        # inverse and determinant
        S_inv = np.linalg.inv(S)
        det_S = np.linalg.det(S)

        # normalization constant
        norm_const = 1 / (((2 * np.pi) ** (d / 2)) * np.sqrt(det_S))

        # Mahalanobis distance (vectorized)
        exp_term = np.sum(diff @ S_inv * diff, axis=1)

        P = norm_const * np.exp(-0.5 * exp_term)

        # numerical stability
        P = np.maximum(P, 1e-300)

        return P

    except Exception:
        return None
