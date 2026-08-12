#!/usr/bin/env python3
"""
Module to create a layer with dropout using TensorFlow.
"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network using dropout.

    Args:
        prev: tensor containing the output of the previous layer
        n: number of nodes the new layer should contain
        activation: activation function for the new layer
        keep_prob: probability that a node will be kept
        training: boolean indicating whether the model is in training mode

    Returns:
        The output tensor of the new layer
    """
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg")
    dense = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init
    )
    layer_output = dense(prev)

    dropout = tf.keras.layers.Dropout(rate=1 - keep_prob)
    return dropout(layer_output, training=training)
