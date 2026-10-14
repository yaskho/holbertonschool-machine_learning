#!/usr/bin/env python3
"""
Module for calculating the derivative of a polynomial represented as a list of coefficients.
"""


def poly_derivative(poly):
    """
    Calculate the derivative of a polynomial.

    Args:
        poly (list): List of coefficients where index represents power of x.

    Returns:
        list: Coefficients of the derivative.
              Returns [0] if the derivative is zero.
              Returns None if input is invalid.
    """
    if not isinstance(poly, list) or not all(isinstance(c, (int, float)) for c in poly):
        return None

    # Derivative: coefficient * power for each term, skip constant term at index 0
    derivative = [i * c for i, c in enumerate(poly)][1:]

    return derivative if derivative else [0]
