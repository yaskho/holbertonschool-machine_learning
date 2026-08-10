#!/usr/bin/env python3
"""
Module to create a hierarchical MultiIndex DataFrame combining bitstamp
and coinbase datasets filtered by timestamp range and sorted chronologically.
"""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Concatenates bitstamp and coinbase tables between timestamps 1417411980
    and 1417417980 (inclusive), with Timestamp as the first level of the
    MultiIndex, sorted chronologically.

    Args:
        df1 (pd.DataFrame): Coinbase DataFrame.
        df2 (pd.DataFrame): Bitstamp DataFrame.

    Returns:
        pd.DataFrame: Concatenated hierarchical DataFrame.
    """
    df1_indexed = index(df1).loc[1417411980:1417417980]
    df2_indexed = index(df2).loc[1417411980:1417417980]

    df = pd.concat([df2_indexed, df1_indexed], keys=['bitstamp', 'coinbase'])
    df = df.swaplevel(0, 1)
    df = df.sort_index()

    return df
