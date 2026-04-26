#!/usr/bin/env python3
"""
Module that makes predictions using a neural network
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using a trained model

    Args:
        network: model used for prediction
        data: input data
        verbose: whether to display progress

    Returns:
        numpy array of predictions
    """
    return network.predict(data, verbose=verbose)
