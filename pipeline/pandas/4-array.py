#!/usr/bin/env python3
"""
Module to convert selected pandas DataFrame columns to a numpy ndarray.
"""


def array(df):
    """
    Selects the last 10 rows of the High and Close columns
    and converts them into a numpy.ndarray.

    Args:
        df (pd.DataFrame): DataFrame containing 'High' and 'Close' columns.

    Returns:
        numpy.ndarray: Array containing the last 10 rows of High and Close.
    """
    return df[['High', 'Close']].tail(10).to_numpy()
