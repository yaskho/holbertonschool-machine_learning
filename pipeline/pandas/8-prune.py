#!/usr/bin/env python3
"""
Module to remove entries where the Close column has NaN values.
"""


def prune(df):
    """
    Removes any entries where Close has NaN values.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    return df.dropna(subset=['Close'])
