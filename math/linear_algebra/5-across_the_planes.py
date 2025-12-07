#!/usr/bin/env python3
"""Module to add two 2D matrices element-wise."""


def add_matrices2D(mat1, mat2):
    """Adds two 2D matrices element-wise.

    Returns a new matrix or None if shapes do not match.
    """
    if len(mat1) != len(mat2) or any(len(r1) != len(r2) for r1, r2 in zip(mat1, mat2)):
        return None
    return [[c1 + c2 for c1, c2 in zip(r1, r2)] for r1, r2 in zip(mat1, mat2)]
