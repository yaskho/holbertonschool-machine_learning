#!/usr/bin/env python3
"""
Creates a TensorFlow inverse time decay learning rate schedule.
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Creates stepwise inverse time decay schedule.

    Parameters
    ----------
    alpha : float
        Initial learning rate.
    decay_rate : float
        Decay factor.
    decay_step : int
        Number of steps before decay occurs.

    Returns
    -------
    tf.keras.optimizers.schedules.LearningRateSchedule
        Learning rate schedule object.
    """
    schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
    return schedule
