#!/usr/bin/env python3
"""Module that determines the definiteness of a matrix."""

import numpy as np


def definiteness(matrix):
    """Calculates the definiteness of a matrix."""

    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.size == 0:
        return None

    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    try:
        eigvals = np.linalg.eigvals(matrix)
    except Exception:
        return None

    # tolérance numérique
    eps = 1e-10

    positive = np.all(eigvals > eps)
    negative = np.all(eigvals < -eps)
    non_negative = np.all(eigvals >= -eps)
    non_positive = np.all(eigvals <= eps)

    if positive:
        return "Positive definite"

    if non_negative and not positive:
        return "Positive semi-definite"

    if negative:
        return "Negative definite"

    if non_positive and not negative:
        return "Negative semi-definite"

    return "Indefinite"
