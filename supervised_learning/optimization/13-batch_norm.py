#!/usr/bin/env python3
"""
Performs batch normalization on a matrix Z.
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes Z using batch normalization.

    Parameters
    ----------
    Z : numpy.ndarray of shape (m, n)
        Input data to normalize.
    gamma : numpy.ndarray of shape (1, n)
        Scale parameter.
    beta : numpy.ndarray of shape (1, n)
        Offset parameter.
    epsilon : float
        Small constant for numerical stability.

    Returns
    -------
    numpy.ndarray
        Batch-normalized Z.
    """
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)

    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    Z_out = gamma * Z_norm + beta

    return Z_out
