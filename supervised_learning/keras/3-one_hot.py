#!/usr/bin/env python3
"""
Converts a label vector into a one-hot encoded matrix.
"""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    One-hot encodes a label vector.

    Args:
        labels (tf.Tensor or list): vector of labels
        classes (int, optional): number of classes

    Returns:
        tf.Tensor: one-hot encoded matrix
    """
    labels = K.backend.cast(labels, 'int32')

    if classes is None:
        classes = K.backend.max(labels) + 1

    return K.backend.one_hot(labels, classes)
