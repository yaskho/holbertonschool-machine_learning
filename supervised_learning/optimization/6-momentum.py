#!/usr/bin/env python3
"""
Creates a TensorFlow optimizer using gradient descent with momentum.
"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Sets up SGD optimizer with momentum.

    Parameters
    ----------
    alpha : float
        Learning rate.
    beta1 : float
        Momentum weight.

    Returns
    -------
    tf.keras.optimizers.Optimizer
        SGD optimizer with momentum.
    """
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=alpha,
        momentum=beta1
    )
    return optimizer
