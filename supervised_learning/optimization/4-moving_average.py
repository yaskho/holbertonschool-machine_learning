#!/usr/bin/env python3
"""
Calculates the weighted moving average of a data set with bias correction.
"""


def moving_average(data, beta):
    """
    Computes moving average using exponential weighting with bias correction.

    Parameters
    ----------
    data : list of float
        Input data.
    beta : float
        Weight for moving average.

    Returns
    -------
    list of float
        Moving averages.
    """
    v = 0
    moving_avgs = []

    for t, x in enumerate(data, 1):
        v = beta * v + (1 - beta) * x
        v_corrected = v / (1 - beta ** t)
        moving_avgs.append(v_corrected)

    return moving_avgs
