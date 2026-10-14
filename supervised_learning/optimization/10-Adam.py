#!/usr/bin/env python3
"""
Creates a TensorFlow Adam optimizer.
"""
import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """
    Sets up Adam optimizer.

    Parameters
    ----------
    alpha : float
        Learning rate.
    beta1 : float
        First moment decay rate.
    beta2 : float
        Second moment decay rate.
    epsilon : float
        Small constant for numerical stability.

    Returns
    -------
    tf.keras.optimizers.Optimizer
        Adam optimizer.
    """
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon
    )
    return optimizer
