#!/usr/bin/env python3
"""
Builds a Keras sequential model with L2 regularization and dropout.
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network using Keras Sequential API.

    Args:
        nx (int): number of input features
        layers (list): number of nodes in each layer
        activations (list): activation functions for each layer
        lambtha (float): L2 regularization parameter
        keep_prob (float): probability a node is kept (dropout)

    Returns:
        keras.Model: compiled Keras sequential model
    """
    model = K.Sequential()

    for i in range(len(layers)):
        # L2 regularization
        regularizer = K.regularizers.l2(lambtha)

        # First layer needs input_dim
        if i == 0:
            model.add(K.layers.Dense(
                units=layers[i],
                activation=activations[i],
                input_shape=(nx,),
                kernel_regularizer=regularizer
            ))
        else:
            model.add(K.layers.Dense(
                units=layers[i],
                activation=activations[i],
                kernel_regularizer=regularizer
            ))

        # Add dropout after every layer
        if i != len(layers) - 1:
            model.add(K.layers.Dropout(rate=1 - keep_prob))

    return model
