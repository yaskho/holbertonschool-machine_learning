#!/usr/bin/env python3
"""
Module for calculating the sum of squares of first n natural numbers.
"""


def summation_i_squared(n):
    """
    Calculate the sum of squares from 1 to n.

    Args:
        n (int): The stopping condition (last integer to sum).

    Returns:
        int: The sum of squares from 1 to n.
        None: If n is not a valid number.
    """
    if not isinstance(n, int) or n < 1:
        return None

    return n * (n + 1) * (2 * n + 1) // 6
