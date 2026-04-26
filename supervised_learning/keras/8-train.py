#!/usr/bin/env python3
"""
Trains a Keras model using mini-batch gradient descent with:
- Early stopping
- Learning rate decay
- Saving best model
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1,
                save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent with optional features.

    Args:
        network (keras.Model): model to train
        data (np.ndarray): input data
        labels (np.ndarray): one-hot labels
        batch_size (int): batch size
        epochs (int): number of epochs
        validation_data (tuple): validation data
        early_stopping (bool): early stopping flag
        patience (int): patience for early stopping
        learning_rate_decay (bool): LR decay flag
        alpha (float): initial learning rate
        decay_rate (float): decay rate
        save_best (bool): save best model flag
        filepath (str): path to save best model
        verbose (bool): print logs
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

    # Save best model
    if save_best and validation_data is not None and filepath is not None:
        callbacks.append(
            K.callbacks.ModelCheckpoint(
                filepath=filepath,
                monitor='val_loss',
                save_best_only=True,
                verbose=verbose
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
