#!/usr/bin/env python3
"""
Trains a Keras model using mini-batch gradient descent with early stopping.
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent with optional early stopping.

    Args:
        network (keras.Model): model to train
        data (np.ndarray): input data (m, nx)
        labels (np.ndarray): one-hot labels (m, classes)
        batch_size (int): size of mini-batch
        epochs (int): number of epochs
        validation_data (tuple): (X_valid, Y_valid)
        early_stopping (bool): whether to use early stopping
        patience (int): patience for early stopping
        verbose (bool): print training progress
        shuffle (bool): shuffle data each epoch

    Returns:
        History: training history object
    """

    callbacks = []

    # Add early stopping if enabled and validation data exists
    if early_stopping and validation_data is not None:
        callbacks.append(
            K.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience
            )
        )

    history = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callbacks
    )

    return history
