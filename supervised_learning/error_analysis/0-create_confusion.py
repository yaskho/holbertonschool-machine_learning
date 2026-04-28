#!/usr/bin/env python3
"""
Creates a confusion matrix
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix

    Parameters:
    labels (numpy.ndarray): one-hot array of shape (m, classes)
    logits (numpy.ndarray): one-hot array of shape (m, classes)

    Returns:
    numpy.ndarray: confusion matrix of shape (classes, classes)
                   rows = true labels, cols = predicted labels
    """
    # Convert one-hot to class indices
    true_classes = np.argmax(labels, axis=1)
    pred_classes = np.argmax(logits, axis=1)

    classes = labels.shape[1]

    # Initialize confusion matrix
    confusion = np.zeros((classes, classes))

    # Fill matrix
    for t, p in zip(true_classes, pred_classes):
        confusion[t][p] += 1

    return confusion
