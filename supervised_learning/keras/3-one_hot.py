#!/usr/bin/env python3
"""
Converts a label vector into a one-hot encoded matrix.
"""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    One-hot encodes a label vector.

    Args:
        labels (array-like): labels vector
        classes (int): number of classes

    Returns:
        tensor: one-hot encoded matrix
    """
    labels = K.backend.cast(labels, 'int32')

    if classes is None:
        classes = K.backend.max(labels) + 1

    return K.backend.one_hot(labels, classes)
