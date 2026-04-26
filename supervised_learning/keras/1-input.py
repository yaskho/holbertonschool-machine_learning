#!/usr/bin/env python3
"""
Builds a Keras model using the Functional API with L2 regularization and dropout.
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network using Keras Functional API.

    Args:
        nx (int): number of input features
        layers (list): number of nodes in each layer
        activations (list): activation functions for each layer
        lambtha (float): L2 regularization parameter
        keep_prob (float): probability a node is kept (dropout)

    Returns:
        keras.Model: constructed Keras model
    """
    inputs = K.Input(shape=(nx,))
    x = inputs

    for i in range(len(layers)):
        regularizer = K.regularizers.l2(lambtha)

        x = K.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=regularizer
        )(x)

        # Add dropout after each layer except last
        if i != len(layers) - 1:
            x = K.layers.Dropout(rate=1 - keep_prob)(x)

    return K.Model(inputs=inputs, outputs=x)
