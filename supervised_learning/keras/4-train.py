#!/usr/bin/env python3
"""
Trains a Keras model using mini-batch gradient descent.
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent.

    Args:
        network (keras.Model): model to train
        data (np.ndarray): input data (m, nx)
        labels (np.ndarray): one-hot labels (m, classes)
        batch_size (int): size of mini-batch
        epochs (int): number of training epochs
        verbose (bool): print training progress
        shuffle (bool): shuffle data each epoch

    Returns:
        History: training history object
    """
    history = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle
    )

    return history
