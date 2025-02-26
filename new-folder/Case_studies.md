# JPMorgan QR Mentorship Program 2024 Case Studies

This repository contains comprehensive solutions for the JPMorgan QR Mentorship Program 2024 Case Studies. The projects cover multiple financial analysis areas, including time series analysis, stock market prediction, and advanced option pricing. Each solution is designed to demonstrate both theoretical knowledge and practical application in financial modeling using Python.

---

## Overview

The case studies are divided into two main projects:

1. **Time Series Analysis & Stock Market Prediction**
   - **Objective:** Analyze historical stock data, compute technical indicators, and forecast future stock prices.
   - **Key Features:**
     - Data acquisition from Yahoo Finance with local caching.
     - Calculation of technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.).
     - Real-time (simulated) visualization of stock price trends and volume.
     - Forecasting future stock prices using an optimized LSTM model with attention, dropout, and batch normalization.
     - Multi-stock analysis using parallel processing.

2. **Option Pricing and Derivative Analysis**
   - **Objective:** Price options and evaluate derivative strategies using analytical and numerical methods.
   - **Key Features:**
     - Black-Scholes analytical pricing for European options with full Greek calculations.
     - Monte Carlo simulation for pricing various option types (European, American, Asian, Lookback) with variance reduction techniques.
     - Binomial tree model for American options.
     - Implied volatility calculation via root-finding methods.
     - Derivative payoff computation for different option strategies.
     - Visualization tools for sensitivity analysis and model interpretability.

---

## Repository Structure

- **/TimeSeries/**  
  Contains code and notebooks related to time series analysis and stock prediction.

- **/OptionPricing/**  
  Contains code for advanced option pricing, including analytical models, Monte Carlo simulations, binomial trees, and derivative analysis.

- **README.md**  
  Provides an overview of the projects, instructions for usage, and details on the methodologies implemented.

---

## Technical Highlights

- **Data Acquisition & Caching:**  
  Utilizes `yfinance` for historical data download with caching to minimize redundant API calls.

- **Technical Indicators:**  
  Computes multiple indicators (moving averages, RSI, MACD, Bollinger Bands, etc.) to analyze market trends.

- **Real-Time Visualization:**  
  Implements simulated real-time charting with dynamic updates and multiple overlays (price, moving averages, volume).

- **Deep Learning Forecasting:**  
  Employs an enhanced LSTM model with attention and regularization techniques for robust price prediction.

- **Option Pricing Models:**  
  Integrates analytical (Black-Scholes) and numerical (Monte Carlo, binomial tree) methods to evaluate option prices and sensitivities.

- **Parallel Processing:**  
  Leverages parallel processing for efficient multi-stock analysis.

- **Visualization & Interpretability:**  
  Provides a suite of plotting functions for technical indicators, model performance, and feature importance analysis.

---

## Usage Instructions

Each project directory includes its own detailed README with step-by-step instructions on running the code, adjusting parameters, and interpreting the outputs. Generally, you can:

1. **Clone the Repository:**
   ```bash
   git clone <repository_url>
   cd <repository_directory>
