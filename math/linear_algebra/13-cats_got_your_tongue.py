#!/usr/bin/env python3
"""Module to concatenate two numpy arrays along a specific axis."""


import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concatenates two numpy arrays along the given axis."""
    return np.concatenate((mat1, mat2), axis=axis)
