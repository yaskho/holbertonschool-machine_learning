#!/usr/bin/env python3
"""
Function that adds two arrays element-wise
"""


def add_arrays(arr1, arr2):
    """
    Adds two arrays element-wise
    Returns a new list, or None if shapes differ
    """
    if len(arr1) != len(arr2):
        return None
    return [arr1[i] + arr2[i] for i in range(len(arr1))]
