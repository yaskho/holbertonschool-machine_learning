#!/usr/bin/env python3
"""
Module to sort a DataFrame in reverse chronological order and transpose it.
"""


def flip_switch(df):
    """
    Sorts the data in reverse chronological order and transposes it.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """
    return df.sort_index(ascending=False).T
