# Enhanced End-to-End Stock Tracker & Predictor

This project provides a comprehensive workflow for time series analysis and stock market prediction using various techniques and models. It includes:

- **Data Acquisition with Caching:** Download historical stock data from Yahoo Finance and cache it locally for efficiency.
- **Technical Indicator Calculation:** Compute various technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.) to analyze stock trends.
- **Enhanced Real-Time Visualization:** Simulated real-time charting with multiple overlays (price, moving averages, volume).
- **Optimized LSTM for Price Prediction:** Train an advanced LSTM model (with attention, dropout, and batch normalization) to forecast future stock prices using multiple features.
- **Derivative Pricing & Sensitivity Analysis:** Calculate option payoff for different strategies and analyze model interpretability using input gradients.
- **Multi-Stock Analysis with Parallel Processing:** Process and analyze multiple stocks concurrently, generating summary statistics and trading signals.
- **Visualization Tools:** Plot price trends, technical indicators, and model outputs to support decision-making.

---

## Table of Contents

- [Overview](#overview)
- [Mathematical and Technical Background](#mathematical-and-technical-background)
  - [Technical Indicators](#technical-indicators)
  - [LSTM for Time Series Prediction](#lstm-for-time-series-prediction)
  - [Derivative Pricing](#derivative-pricing)
- [Library Modules](#library-modules)
  - [Data Acquisition & Caching](#data-acquisition--caching)
  - [Technical Indicator Calculation](#technical-indicator-calculation)
  - [Real-Time Visualization](#real-time-visualization)
  - [LSTM Model & Training](#lstm-model--training)
  - [Derivative Payoff & Model Interpretability](#derivative-payoff--model-interpretability)
  - [Multi-Stock Analysis](#multi-stock-analysis)
  - [Visualization Tools](#visualization-tools)
- [Usage Examples](#usage-examples)
- [Enhanced Considerations](#enhanced-considerations)
- [Conclusion](#conclusion)

---

## Overview

Time series data is ubiquitous in finance. This project focuses on analyzing stock market data, computing technical indicators, forecasting future prices with an optimized LSTM model, and assessing derivative payoffs. The solution integrates real-time simulation, parallel processing for multi-stock analysis, and advanced visualization tools.

---

## Mathematical and Technical Background

### Technical Indicators

- **Simple Moving Average (SMA):**  
  Average of a set of prices over a specific period (e.g., 20-day, 50-day, 200-day).

- **Exponential Moving Average (EMA):**  
  Similar to SMA but gives more weight to recent prices.

- **Relative Strength Index (RSI):**  
  Measures the speed and change of price movements to identify overbought or oversold conditions.

- **MACD (Moving Average Convergence Divergence):**  
  Difference between two EMAs (typically 12-day and 26-day) with a signal line (usually 9-day EMA).

- **Bollinger Bands:**  
  Consist of a middle band (SMA) and upper/lower bands set at a fixed number of standard deviations from the middle band.

### LSTM for Time Series Prediction

The Long Short-Term Memory (LSTM) network is a type of recurrent neural network (RNN) that can learn long-term dependencies. Key features in our model include:
- **Dropout:** Prevents overfitting by randomly dropping connections during training.
- **Batch Normalization:** Improves training stability.
- **Attention Mechanism (Simplified):** Weighs different time steps of the input sequence to enhance prediction quality.
- **Multi-Feature Input:** Uses not only closing prices but additional indicators (Volume, SMA, RSI, MACD) to improve forecasting accuracy.

### Derivative Pricing

Derivative pricing in this project includes:
- **Standard Option Payoffs:**  
  For calls and puts, the payoff is calculated as:
  - Call: \( \max(S_{\text{pred}} - K, 0) \)
  - Put: \( \max(K - S_{\text{pred}}, 0) \)
- **Spreads:**  
  For call and put spreads, differences between payoffs at two strike levels are computed.
- **Return on Investment (ROI):**  
  Calculated if a premium is provided.

---

## Library Modules

### Data Acquisition & Caching

- **`fetch_stock_data(ticker, period, interval, use_cache)`**  
  Downloads historical data using `yfinance` and caches it locally to improve efficiency.

### Technical Indicator Calculation

- **`add_technical_indicators(df)`**  
  Computes technical indicators like SMA (20, 50, 200), EMA (12, 26), MACD (and its signal/histogram), RSI, Bollinger Bands, Volume SMA, and Momentum.

### Real-Time Visualization

- **`real_time_plot(df, ticker)`**  
  Creates a simulated real-time updating chart with price data, moving averages, and volume bars using matplotlib's `FuncAnimation`.

### LSTM Model & Training

- **`EnhancedLSTMPredictor`**  
  An LSTM model with dropout, batch normalization, and a simple attention mechanism.
- **`prepare_data(df, features, seq_length, test_size)`** and **`create_sequences(...)`**  
  Prepare and convert the data for LSTM training.
- **`train_lstm(df, features, seq_length, epochs, lr, batch_size, patience)`**  
  Trains the LSTM model with early stopping, learning rate scheduling, and batch processing. It also evaluates the model with metrics like MSE, RMSE, MAE, and R².
- **`predict_future(model, scaler, recent_data, features, seq_length, future_steps)`**  
  Uses the trained LSTM to forecast future stock prices over a given number of steps.

### Derivative Payoff & Model Interpretability

- **`derivative_payoff(predicted_price, strike_price, option_type, premium)`**  
  Calculates the payoff and ROI for various option types (call, put, call spread, put spread).
- **`interpret_model(model, sample_sequence, target_idx)`**  
  Computes input gradients for enhanced model interpretability.
- **`visualize_feature_importance(feature_importance, df, seq_length)`**  
  Visualizes which time steps in the input sequence most influence the prediction.

### Multi-Stock Analysis

- **`analyze_multiple_stocks(tickers, period, interval)`**  
  Uses parallel processing to analyze multiple stocks concurrently, computing basic statistics and technical signals for each ticker.

### Visualization Tools

- **Price and Moving Averages Plot:**  
  Plots stock closing prices along with SMA lines and Bollinger Bands.
- **RSI, MACD Plots:**  
  Visualizes RSI and MACD values to assist in technical analysis.

---

## Usage Examples

Below are some usage examples included in the `__main__` block of the code:

- **Data Acquisition & Indicator Calculation:**
  ```python
  df = fetch_stock_data("AAPL", period="1y", interval="1d")
  df = add_technical_indicators(df)
  print(df.head())
