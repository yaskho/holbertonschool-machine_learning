#!/usr/bin/env python3
"""
Module to sort a DataFrame by High price in descending order.
"""


def high(df):
    """
    Sorts a DataFrame by the High price in descending order.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The sorted DataFrame.
    """
    return df.sort_values(by='High', ascending=False)
