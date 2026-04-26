#!/usr/bin/env python3
import numpy as np


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix.

    Parameters
    ----------
    X : numpy.ndarray of shape (d, nx)
        Matrix to normalize.
    m : numpy.ndarray of shape (nx,)
        Mean of each feature.
    s : numpy.ndarray of shape (nx,)
        Standard deviation of each feature.

    Returns
    -------
    numpy.ndarray
        The normalized matrix.
    """
    X_norm = (X - m) / s
    return X_norm
