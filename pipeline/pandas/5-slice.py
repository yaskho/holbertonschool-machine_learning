#!/usr/bin/env python3
"""
Module to slice specific columns and rows from a pandas DataFrame.
"""


def slice(df):
    """
    Extracts High, Low, Close, and Volume_(BTC) columns and
    selects every 60th row.

    Args:
        df (pd.DataFrame): DataFrame containing crypto data.

    Returns:
        pd.DataFrame: Sliced DataFrame containing selected columns and rows.
    """
    return df[['High', 'Low', 'Close', 'Volume_(BTC)']].iloc[::60]
