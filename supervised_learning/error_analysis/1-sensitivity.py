#!/usr/bin/env python3
"""Module for calculating sensitivity"""
import numpy as np


def sensitivity(confusion):
    """Calculates the sensitivity for each class

    Args:
        confusion (numpy.ndarray): shape (classes, classes)

    Returns:
        numpy.ndarray: sensitivity for each class
    """
    TP = np.diag(confusion)
    actual = np.sum(confusion, axis=1)
    return TP / actual
