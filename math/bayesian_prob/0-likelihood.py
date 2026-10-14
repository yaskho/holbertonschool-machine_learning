#!/usr/bin/env python3
"""Likelihood module."""

import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining the data.

    Args:
        x (int): number of patients with severe side effects
        n (int): total number of patients
        P (numpy.ndarray): hypothetical probabilities

    Returns:
        numpy.ndarray: likelihood for each probability in P
    """

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any(P < 0) or np.any(P > 1):
        raise ValueError(
            "All values in P must be in the range [0, 1]"
        )

    factorial_n = np.math.factorial(n)
    factorial_x = np.math.factorial(x)
    factorial_nx = np.math.factorial(n - x)

    combination = factorial_n / (factorial_x * factorial_nx)

    return combination * (P ** x) * ((1 - P) ** (n - x))
