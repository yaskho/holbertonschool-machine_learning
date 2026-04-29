#!/usr/bin/env python3
"""Module for creating a layer with L2 regularization"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Creates a dense layer with L2 regularization

    Args:
        prev (tensor): output of previous layer
        n (int): number of nodes
        activation: activation function
        lambtha (float): L2 regularization parameter

    Returns:
        tensor: output of the new layer
    """
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode="fan_avg"
    )

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=tf.keras.regularizers.L2(lambtha)
    )

    return layer(prev)
