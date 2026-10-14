#!/usr/bin/env python3
"""Bitcoin price forecasting using Recurrent Neural Networks in TensorFlow.

Loads preprocessed hourly BTC data, builds a tf.data sliding window input
pipeline (24 hours past -> predict next hour close), trains an LSTM/GRU
architecture, and evaluates performance using Mean Squared Error (MSE).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


def load_and_preprocess_data(csv_path="preprocessed_btc.csv"):
    """Load preprocessed CSV data and split into train, val, and test sets.

    Args:
        csv_path (str): Path to preprocessed CSV file.

    Returns:
        tuple: (train_data, val_data, test_data, scaler)
    """
    df = pd.read_csv(csv_path)

    # Select numerical feature columns
    feature_cols = ['Open', 'High', 'Low', 'Close',
                    'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price']
    data = df[feature_cols].values

    # Chronological split: 70% train, 20% validation, 10% test
    n = len(data)
    train_data = data[:int(n * 0.7)]
    val_data = data[int(n * 0.7):int(n * 0.9)]
    test_data = data[int(n * 0.9):]

    # Fit scaler ONLY on training set to avoid data leakage
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)

    return train_scaled, val_scaled, test_scaled, scaler


def create_windowed_dataset(data, window_size=24, batch_size=32,
                            shuffle=True):
    """Create tf.data.Dataset using sliding windows.

    Uses past 24 hours of features (X) to predict the close price
    at time t + 1 (y, index 3 corresponds to 'Close').

    Args:
        data (np.ndarray): Scaled feature array of shape (N, num_features).
        window_size (int): Number of historical time steps (default 24).
        batch_size (int): Batch size for training (default 32).
        shuffle (bool): Whether to shuffle dataset samples.

    Returns:
        tf.data.Dataset: Batched and prefetched dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices(data)

    # Extract windows of window_size + 1 (24 input steps + 1 target)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)

    # Flatten nested dataset windows into batches
    dataset = dataset.flat_map(lambda w: w.batch(window_size + 1))

    # X: past 24 time steps (all features), y: next hour 'Close' (index 3)
    dataset = dataset.map(lambda w: (w[:-1, :], w[-1, 3]))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def build_rnn_model(input_shape):
    """Build and compile stacked RNN model for BTC forecasting.

    Args:
        input_shape (tuple): Shape of input sequences (time_steps, features).

    Returns:
        tf.keras.Model: Compiled Keras Sequential model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units=64, return_sequences=True,
                             input_shape=input_shape),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GRU(units=32, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=16, activation='relu'),
        tf.keras.layers.Dense(units=1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    return model


def train_and_evaluate():
    """Execute complete training and evaluation pipeline."""
    train_s, val_s, test_s, scaler = load_and_preprocess_data()

    window_size = 24
    batch_size = 32

    # Prepare TensorFlow input pipelines
    train_ds = create_windowed_dataset(train_s, window_size, batch_size,
                                       shuffle=True)
    val_ds = create_windowed_dataset(val_s, window_size, batch_size,
                                     shuffle=False)
    test_ds = create_windowed_dataset(test_s, window_size, batch_size,
                                      shuffle=False)

    num_features = train_s.shape[1]
    input_shape = (window_size, num_features)

    model = build_rnn_model(input_shape)
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        train_ds,
        epochs=20,
        validation_data=val_ds,
        callbacks=[early_stop]
    )

    test_loss, test_mae = model.evaluate(test_ds)
    print(f"Test MSE Loss: {test_loss:.6f}")
    print(f"Test MAE: {test_mae:.6f}")

    model.save("btc_forecast_model.h5")
    print("Model saved to btc_forecast_model.h5")


if __name__ == "__main__":
    train_and_evaluate()
