#!/usr/bin/env python3
"""
Module that tests a neural network model
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network

    Args:
        network: model to test
        data: input data
        labels: correct one-hot labels
        verbose: whether to display progress

    Returns:
        loss, accuracy
    """
    return network.evaluate(data, labels, verbose=verbose)
