#!/usr/bin/env python3
"""
Module to index and concatenate two pandas DataFrames with hierarchical keys.
"""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
    Indexes two dataframes on Timestamp, filters df2 up to timestamp
    1417411920, and concatenates df2 on top of df1 with hierarchical keys.

    Args:
        df1 (pd.DataFrame): Coinbase DataFrame.
        df2 (pd.DataFrame): Bitstamp DataFrame.

    Returns:
        pd.DataFrame: Concatenated DataFrame labeled with keys 'bitstamp'
                      and 'coinbase'.
    """
    df1_indexed = index(df1)
    df2_indexed = index(df2)

    df2_filtered = df2_indexed.loc[:1417411920]

    return pd.concat([df2_filtered, df1_indexed], keys=['bitstamp', 'coinbase'])
