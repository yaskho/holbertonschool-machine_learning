#!/usr/bin/env python3
"""
Module to rename columns and convert timestamps to datetime objects.
"""
import pandas as pd


def rename(df):
    """
    Renames the Timestamp column to Datetime, converts Unix timestamps
    to datetime objects, and returns only the Datetime and Close columns.

    Args:
        df (pd.DataFrame): DataFrame containing a 'Timestamp' column.

    Returns:
        pd.DataFrame: Modified DataFrame with only 'Datetime' and 'Close'.
    """
    df = df.rename(columns={'Timestamp': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    df = df[['Datetime', 'Close']]
    return df
