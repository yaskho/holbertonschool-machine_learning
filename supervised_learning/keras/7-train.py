#!/usr/bin/env python3
"""
Trains a Keras model using mini-batch gradient descent with
early stopping and learning rate decay.
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent with:
    - Early stopping (optional)
    - Learning rate decay (optional)

    Args:
        network (keras.Model): model to train
        data (np.ndarray): input data
        labels (np.ndarray): one-hot labels
        batch_size (int): batch size
        epochs (int): number of epochs
        validation_data (tuple): validation data
        early_stopping (bool): whether to use early stopping
        patience (int): patience for early stopping
        learning_rate_decay (bool): whether to use LR decay
        alpha (float): initial learning rate
        decay_rate (float): decay rate
        verbose (bool): print training progress
        shuffle (bool): shuffle data each epoch

    Returns:
        History: training history object
    """

    callbacks = []

    # Early stopping
    if early_stopping and validation_data is not None:
        callbacks.append(
            K.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience
            )
        )

    # Learning rate decay (inverse time decay)
    def schedule(epoch):
        return alpha / (1 + decay_rate * epoch)

    if learning_rate_decay and validation_data is not None:
        callbacks.append(
            K.callbacks.LearningRateScheduler(schedule, verbose=1)
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
