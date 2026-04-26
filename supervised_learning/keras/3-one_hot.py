#!/usr/bin/env python3
"""
Converts a label vector into a one-hot encoded matrix.
"""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    One-hot encodes a label vector.

    Args:
        labels (tf.Tensor or np.ndarray): vector of labels
        classes (int, optional): number of classes

    Returns:
        tf.Tensor: one-hot encoded matrix
    """
    labels = K.backend.constant(labels)

    if classes is None:
        classes = K.backend.max(labels) + 1

    one_hot_matrix = K.backend.one_hot(labels, int(classes))

    return one_hot_matrix
