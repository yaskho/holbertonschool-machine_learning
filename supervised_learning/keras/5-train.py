#!/usr/bin/env python3
"""
Trains a Keras model using mini-batch gradient descent with validation data.
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent.

    Args:
        network (keras.Model): model to train
        data (np.ndarray): input data (m, nx)
        labels (np.ndarray): one-hot labels (m, classes)
        batch_size (int): size of mini-batch
        epochs (int): number of training epochs
        validation_data (tuple): (X_valid, Y_valid), optional
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
        shuffle=shuffle,
        validation_data=validation_data
    )

    return history
