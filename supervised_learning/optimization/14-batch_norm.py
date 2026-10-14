#!/usr/bin/env python3
"""
Creates a batch normalization layer for a neural network.
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer.

    Parameters
    ----------
    prev : tensor
        Activated output of previous layer.
    n : int
        Number of nodes in the layer.
    activation : callable
        Activation function.

    Returns
    -------
    tensor
        Activated output of the batch normalization layer.
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # Dense layer without activation
    Z = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=initializer
    )(prev)

    # Batch normalization parameters
    gamma = tf.Variable(tf.ones((1, n)), trainable=True)
    beta = tf.Variable(tf.zeros((1, n)), trainable=True)

    # Compute batch statistics
    mean, variance = tf.nn.moments(Z, axes=[0])

    epsilon = 1e-7

    Z_norm = tf.nn.batch_normalization(
        Z,
        mean,
        variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=epsilon
    )

    # Apply activation
    return activation(Z_norm)
