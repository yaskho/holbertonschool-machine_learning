#!/usr/bin/env python3
"""
Creates a TensorFlow RMSProp optimizer.
"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Sets up RMSProp optimizer.

    Parameters
    ----------
    alpha : float
        Learning rate.
    beta2 : float
        RMSProp decay rate.
    epsilon : float
        Small constant for numerical stability.

    Returns
    -------
    tf.keras.optimizers.Optimizer
        RMSProp optimizer.
    """
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )
    return optimizer
