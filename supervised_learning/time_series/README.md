Markdown
# Time Series Forecasting - Bitcoin (BTC) Price Prediction

## Project Overview
This project builds a deep learning time series forecasting pipeline in TensorFlow to predict the value of Bitcoin (BTC) at the close of the following hour based on historical trading data from the past 24 hours.

## Preprocessing Design Choices
1. **Time Aggregation**: Minute-level resolution (1,440 time steps per day) introduces significant noise and unnecessary computational overhead. Aggregating raw 60-second transactions into **1-hour OHLCV intervals** lowers the sequence length to 24 time steps for 24 hours of context.
2. **Feature Selection**: All primary features (`Open`, `High`, `Low`, `Close`, `Volume_(BTC)`, `Volume_(Currency)`, `Weighted_Price`) are retained as multivariate inputs to enhance temporal feature representation.
3. **Data Cleaning**: Early data (prior to 2017) containing sparse transactions and large missing windows is removed. Missing internal price points are forward-filled (`ffill`).
4. **Data Scaling**: `StandardScaler` is fit **exclusively** on the training set split to prevent target data leakage into validation/test splits.

## Model Architecture
- **Input Pipeline**: Built using `tf.data.Dataset.window()` to form overlapping sliding windows of length 24 $\rightarrow$ target step 1.
- **Layers**: Stacked `LSTM` (64 units) and `GRU` (32 units) layers with `Dropout` (0.2) to mitigate overfitting, followed by dense output heads.
- **Loss Function**: Mean Squared Error (`MSE`).

## Requirements
- OS: Ubuntu 20.04 LTS
- Python: 3.9
- TensorFlow: 2.15
- NumPy: 1.25.2
- Pandas: 2.2.2

## Execution Instructions
Make the scripts executable:
```bash
chmod +x preprocess_data.py forecast_btc.py