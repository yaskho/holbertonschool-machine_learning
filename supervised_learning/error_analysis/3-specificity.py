#!/usr/bin/env python3
"""Module for calculating specificity"""
import numpy as np


def specificity(confusion):
    """Calculates the specificity for each class

    Args:
        confusion (numpy.ndarray): shape (classes, classes)

    Returns:
        numpy.ndarray: specificity for each class
    """
    total = np.sum(confusion)

    TP = np.diag(confusion)
    FP = np.sum(confusion, axis=0) - TP
    FN = np.sum(confusion, axis=1) - TP
    TN = total - (TP + FP + FN)

    return TN / (TN + FP)
