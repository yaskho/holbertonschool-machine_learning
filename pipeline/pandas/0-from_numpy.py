#!/usr/bin/env python3
"""
Module to create a pandas DataFrame from a numpy ndarray.
"""
import pandas as pd


def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray with capitalized
    alphabetical column labels.

    Args:
        array (np.ndarray): The numpy array to convert.

    Returns:
        pd.DataFrame: The newly created DataFrame.
    """
    num_cols = array.shape[1]
    columns = [chr(ord('A') + i) for i in range(num_cols)]
    return pd.DataFrame(array, columns=columns)
