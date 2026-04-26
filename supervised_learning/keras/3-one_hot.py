#!/usr/bin/env python3
"""
Converts a label vector into a one-hot encoded matrix.
"""

import numpy as np


def one_hot(labels, classes=None):
    """
    One-hot encodes a label vector.

    Args:
        labels (np.ndarray): vector of labels
        classes (int, optional): number of classes

    Returns:
        np.ndarray: one-hot encoded matrix
    """
    labels = np.array(labels)

    if classes is None:
        classes = np.max(labels) + 1

    one_hot_matrix = np.eye(classes)[labels]

    return one_hot_matrix
