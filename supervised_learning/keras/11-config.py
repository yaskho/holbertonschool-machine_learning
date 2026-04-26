#!/usr/bin/env python3
"""
Module that saves and loads a model's configuration
"""

import tensorflow.keras as K
import json


def save_config(network, filename):
    """
    Saves a model's configuration in JSON format

    Args:
        network: model whose configuration should be saved
        filename: file path to save configuration

    Returns:
        None
    """
    config = network.to_json()
    with open(filename, "w") as f:
        f.write(config)


def load_config(filename):
    """
    Loads a model with a specific configuration

    Args:
        filename: JSON file containing model configuration

    Returns:
        The loaded Keras model
    """
    with open(filename, "r") as f:
        config = f.read()

    return K.models.model_from_json(config)
