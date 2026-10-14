#!/usr/bin/env python3
"""Preprocessing script for Bitcoin time series forecasting data.

Cleans raw minute-level Bitcoin transaction data from Bitstamp/Coinbase CSV
files, resamples to 1-hour intervals, fills missing values, and exports the
preprocessed dataset.
"""

import numpy as np
import pandas as pd


def preprocess_data(file_path, output_path="preprocessed_btc.csv"):
    """Clean and resample raw BTC minute data into 1-hour intervals.

    Args:
        file_path (str): Path to raw CSV file.
        output_path (str): Path for saving preprocessed CSV output.

    Returns:
        pd.DataFrame: Processed hourly DataFrame.
    """
    df = pd.read_csv(file_path)

    # Convert Unix timestamp to UTC datetime index
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df.set_index('Timestamp', inplace=True)

    # Filter out early incomplete/sparse data prior to 2017
    df = df.loc['2017-01-01':]

    # Handle missing values in price and volume columns
    df['Close'] = df['Close'].ffill()
    df['Open'] = df['Open'].fillna(df['Close'])
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)
    df['Weighted_Price'] = df['Weighted_Price'].fillna(df['Close'])

    # Aggregate minute-level data to 1-hour OHLCV intervals
    resampled = df.resample('1h').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume_(BTC)': 'sum',
        'Volume_(Currency)': 'sum',
        'Weighted_Price': 'mean'
    })

    # Fill any remaining empty hourly slots
    resampled = resampled.ffill().bfill()

    # Export to CSV
    resampled.to_csv(output_path)
    print(f"Preprocessed data saved successfully to {output_path}")
    return resampled


if __name__ == "__main__":
    import sys

    raw_file = "coinbaseUSD_1-min_data_2012-01-01_to_2019-02-28.csv"
    if len(sys.argv) > 1:
        raw_file = sys.argv[1]

    preprocess_data(raw_file)
