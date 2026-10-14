#!/usr/bin/env python3
"""t-SNE entropy module."""

import numpy as np


def HP(Di, beta):
    """
    Calculates Shannon entropy and P affinities.

    Args:
        Di (np.ndarray): shape (n - 1,), pairwise distances
        beta (np.ndarray): shape (1,), precision of Gaussian

    Returns:
        Hi (float): Shannon entropy
        Pi (np.ndarray): shape (n - 1,), affinities
    """

    beta = float(beta)

    # Compute unnormalized probabilities
    Pi = np.exp(-Di * beta)

    # Normalize
    sum_Pi = np.sum(Pi)
    Pi = Pi / sum_Pi

    # Avoid log(0)
    Pi_safe = np.where(Pi == 0, 1e-10, Pi)

    # Shannon entropy (base 2)
    Hi = -np.sum(Pi * np.log2(Pi_safe))

    return Hi, Pi
