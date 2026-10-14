#!/usr/bin/env python3
"""
Updates learning rate using inverse time step decay.
"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Computes stepwise learning rate decay.

    Parameters
    ----------
    alpha : float
        Initial learning rate.
    decay_rate : float
        Decay factor.
    global_step : int
        Number of gradient descent steps so far.
    decay_step : int
        Steps before applying decay.

    Returns
    -------
    float
        Updated learning rate.
    """
    step = global_step // decay_step
    alpha_decayed = alpha / (1 + decay_rate * step)
    return alpha_decayed
