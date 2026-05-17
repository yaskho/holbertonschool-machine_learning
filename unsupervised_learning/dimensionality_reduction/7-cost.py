#!/usr/bin/env python3
"""t-SNE cost function module."""

import numpy as np


def cost(P, Q):
    """
    Calculates t-SNE cost (KL divergence between P and Q).

    Args:
        P (np.ndarray): high-dimensional affinities (n, n)
        Q (np.ndarray): low-dimensional affinities (n, n)

    Returns:
        float: cost value
    """

    eps = 1e-12

    P_safe = np.maximum(P, eps)
    Q_safe = np.maximum(Q, eps)

    C = np.sum(P_safe * np.log(P_safe / Q_safe))

    return C
