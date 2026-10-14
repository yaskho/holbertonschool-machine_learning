#!/usr/bin/env python3
"""
Module to set the Timestamp column as the index of a pandas DataFrame.
"""


def index(df):
    """
    Sets the Timestamp column as the index of the dataframe.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The modified DataFrame with Timestamp as index.
    """
    return df.set_index('Timestamp')
