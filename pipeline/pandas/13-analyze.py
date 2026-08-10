#!/usr/bin/env python3
"""
Module to compute descriptive statistics for a pandas DataFrame.
"""


def analyze(df):
    """
    Computes descriptive statistics for all columns except 'Timestamp'.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing descriptive statistics for
                      all columns excluding Timestamp.
    """
    return df.drop(columns=['Timestamp']).describe()
