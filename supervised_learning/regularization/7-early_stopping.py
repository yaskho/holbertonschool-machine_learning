#!/usr/bin/env python3
"""
Module to evaluate early stopping in neural network training.
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if gradient descent should be stopped early.

    Args:
        cost: current validation cost
        opt_cost: lowest recorded validation cost
        threshold: threshold used for early stopping
        patience: patience count used for early stopping
        count: count of how long the threshold has not been met

    Returns:
        tuple: (boolean indicating whether to stop early, updated count)
    """
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1

    return count >= patience, count
