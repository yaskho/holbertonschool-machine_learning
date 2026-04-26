#!/usr/bin/env python3
"""
Model saving and loading functions
"""

import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire model to a file.

    Args:
        network: model to save
        filename: path to save the model

    Returns:
        None
    """
    network.save(filename)


def load_model(filename):
    """
    Loads an entire model from a file.

    Args:
        filename: path of the saved model

    Returns:
        The loaded Keras model
    """
    return K.models.load_model(filename)
