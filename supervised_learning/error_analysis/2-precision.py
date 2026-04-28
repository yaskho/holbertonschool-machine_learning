#!/usr/bin/env python3
"""Module for calculating precision"""
import numpy as np


def precision(confusion):
    """Calculates the precision for each class

    Args:
        confusion (numpy.ndarray): shape (classes, classes)

    Returns:
        numpy.ndarray: precision for each class
    """
    TP = np.diag(confusion)
    predicted = np.sum(confusion, axis=0)
    return TP / predicted
