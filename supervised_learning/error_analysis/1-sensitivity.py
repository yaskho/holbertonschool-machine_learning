#!/usr/bin/env python3
"""
Calculates the sensitivity (recall) for each class
"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class

    Parameters:
    confusion (numpy.ndarray): shape (classes, classes)

    Returns:
    numpy.ndarray: shape (classes,) containing sensitivity per class
    """
    # True Positives (diagonal)
    TP = np.diag(confusion)

    # Total actual per class (row sum)
    actual = np.sum(confusion, axis=1)

    # Sensitivity = TP / (TP + FN)
    sensitivity = TP / actual

    return sensitivity
