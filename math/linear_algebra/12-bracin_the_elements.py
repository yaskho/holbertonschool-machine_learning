#!/usr/bin/env python3
"""Module to perform element-wise operations on numpy arrays."""


def np_elementwise(mat1, mat2):
    """Performs element-wise addition, subtraction, multiplication, and division.
    
    Returns a tuple: (sum, difference, product, quotient).
    """
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2
    return add, sub, mul, div
