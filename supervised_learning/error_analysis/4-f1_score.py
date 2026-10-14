#!/usr/bin/env python3
"""Module for calculating F1 score"""
import numpy as np

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """Calculates the F1 score for each class

    Args:
        confusion (numpy.ndarray): shape (classes, classes)

    Returns:
        numpy.ndarray: F1 score for each class
    """
    prec = precision(confusion)
    rec = sensitivity(confusion)

    return 2 * (prec * rec) / (prec + rec)
