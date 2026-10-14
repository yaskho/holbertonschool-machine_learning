#!/usr/bin/env python3
"""
Module to fill missing values in a crypto dataset DataFrame.
"""


def fill(df):
    """
    Fills missing values in a pandas DataFrame according to specific rules:
    - Removes 'Weighted_Price' column.
    - Fills missing 'Close' values with the previous row's value.
    - Fills missing 'High', 'Low', and 'Open' values with the 'Close' value.
    - Sets missing 'Volume_(BTC)' and 'Volume_(Currency)' values to 0.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    df = df.drop(columns=['Weighted_Price'])

    df['Close'] = df['Close'].ffill()

    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Open'] = df['Open'].fillna(df['Close'])

    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

    return df
