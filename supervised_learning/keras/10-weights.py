#!/usr/bin/env python3
"""
Module that saves and loads a model's weights
"""


def save_weights(network, filename, save_format='keras'):
    """
    Saves a model's weights to a file

    Args:
        network: the model whose weights should be saved
        filename: path of the file to save weights to
        save_format: format in which weights are saved

    Returns:
        None
    """
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """
    Loads a model's weights from a file

    Args:
        network: the model to load weights into
        filename: path of the weights file

    Returns:
        None
    """
    network.load_weights(filename)
