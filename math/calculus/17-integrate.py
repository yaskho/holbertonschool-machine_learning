#!/usr/bin/env python3
"""
Module for calculating the integral of a polynomial represented as a list of coefficients.
"""


def poly_integral(poly, C=0):
    """
    Calculate the integral of a polynomial.

    Args:
        poly (list): List of coefficients where index represents the power of x.
        C (int, optional): Integration constant. Defaults to 0.

    Returns:
        list: Coefficients of the integrated polynomial.
              Returns None if input is invalid.
    """
    if not isinstance(poly, list) or not all(isinstance(c, (int, float)) for c in poly):
        return None
    if not isinstance(C, int):
        return None

    integral = [C]
    for power, coeff in enumerate(poly):
        new_coeff = coeff / (power + 1)
        integral.append(int(new_coeff) if new_coeff.is_integer() else new_coeff)

    # Remove trailing zeros to make list as small as possible
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
