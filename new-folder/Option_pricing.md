# Enhanced Option Pricing Case study 

This repository provides a comprehensive Python library for option pricing and analysis. It includes multiple methods for pricing options, calculating sensitivities (Greeks), and validating models with real market data. The library also offers visualization tools for sensitivity analysis and option price surfaces.

The key functionalities include:

1. **Black-Scholes Analytical Pricing**:  
   - Pricing for European options (calls and puts).  
   - Calculation of first and second order Greeks (Delta, Gamma, Vega, Theta, Rho, Charm, Vanna, Vomma).

2. **Monte Carlo Simulation**:  
   - Simulation for various option types (European, American, Asian, and Lookback).  
   - Variance reduction techniques such as antithetic variates and control variates.  
   - Returns both the estimated option price and an error estimate (confidence interval).

3. **Binomial Tree Model**:  
   - Pricing for American options using the Cox-Ross-Rubinstein model.

4. **Implied Volatility Calculation**:  
   - Uses Brent’s method to compute implied volatility for a given market price.

5. **Market Data Integration**:  
   - Functions to retrieve market data (stock prices, risk-free rates, dividend yields) and option chains using `yfinance`.

6. **Visualization Tools**:  
   - Plotting option price surfaces, Greeks as functions of the underlying price, and implied volatility smiles from market data.

---

## Table of Contents

- [Overview](#overview)
- [Mathematical Background](#mathematical-background)
  - [Black-Scholes Model](#black-scholes-model)
  - [Option Greeks](#option-greeks)
  - [Monte Carlo Simulation](#monte-carlo-simulation)
- [Library Modules](#library-modules)
  - [Black-Scholes Pricing & Greeks](#black-scholes-pricing--greeks)
  - [Monte Carlo Simulation](#monte-carlo-simulation)
  - [Binomial Tree Model](#binomial-tree-model)
  - [Implied Volatility](#implied-volatility)
  - [Market Data Integration](#market-data-integration)
  - [Visualization Tools](#visualization-tools)
- [Usage Examples](#usage-examples)
- [Enhanced Considerations](#enhanced-considerations)
- [Conclusion](#conclusion)

---

## Overview

This library is designed to bridge the gap between theoretical option pricing models and practical, real-world application. It includes:

- **Analytical models (Black-Scholes)** for fast pricing and sensitivity analysis.
- **Numerical methods (Monte Carlo and Binomial Trees)** for pricing more complex options and for validation against analytical results.
- **Market data integration** to facilitate model validation with real-time data.
- **Visualization tools** to analyze sensitivity (Greeks) and the option price landscape.

---

## Mathematical Background

### Black-Scholes Model

The Black-Scholes formula for a European call option is:

\[
C = S\,e^{-qT}N(d_1) - K\,e^{-rT}N(d_2)
\]

Where:
- \( S \) is the current stock price.
- \( K \) is the strike price.
- \( T \) is the time to expiration (in years).
- \( r \) is the risk-free rate (annualized).
- \( \sigma \) is the volatility (annualized).
- \( q \) is the continuous dividend yield.
- \( N(\cdot) \) is the cumulative standard normal distribution.
- \( d_1 \) and \( d_2 \) are computed as:

\[
d_1 = \frac{\ln(S/K) + (r - q + 0.5\sigma^2)T}{\sigma \sqrt{T}}, \quad
d_2 = d_1 - \sigma \sqrt{T}
\]

### Option Greeks

The Greeks measure sensitivities:
- **Delta (\(\Delta\))**: \( e^{-qT}N(d_1) \) for calls (adjusted for puts accordingly).
- **Gamma (\(\Gamma\))**: \( \frac{e^{-qT}N'(d_1)}{S\sigma\sqrt{T}} \).
- **Vega**: \( S\,e^{-qT}\sqrt{T}\,N'(d_1) \).
- **Theta**: Represents time decay (different formula for calls and puts).
- **Rho**: Sensitivity to the interest rate.
- **Charm, Vanna, Vomma**: Second-order sensitivities for more advanced analysis.

### Monte Carlo Simulation

Monte Carlo simulation estimates the expected payoff under the risk-neutral measure:

\[
S_T = S_0 \exp\!\left\{ (r - q - \tfrac{1}{2}\sigma^2)T + \sigma\sqrt{T}\,Z \right\}
\]

Where \( Z \sim N(0,1) \).  
The option price is given by discounting the expected payoff:

\[
C = e^{-rT} \mathbb{E}\left[\max(S_T - K, 0)\right]
\]

Variance reduction techniques like **antithetic variates** and **control variates** are implemented for improved convergence.

---

## Library Modules

### Black-Scholes Pricing & Greeks

- **`black_scholes_price(S, K, T, r, sigma, option_type, q)`**:  
  Calculates the price of a European option using the Black-Scholes model.

- **`black_scholes_greeks(S, K, T, r, sigma, option_type, q)`**:  
  Returns a dictionary of Greeks (Delta, Gamma, Vega, Theta, Rho, Charm, Vanna, Vomma).

### Monte Carlo Simulation

- **`monte_carlo_option_price(...)`**:  
  Prices options via simulation for multiple option styles (European, Asian, Lookback, American) with options for variance reduction.

### Binomial Tree Model

- **`binomial_tree_price(S, K, T, r, sigma, option_type, option_style, q, n_steps)`**:  
  Prices European or American options using a Cox-Ross-Rubinstein binomial tree.

### Implied Volatility

- **`implied_volatility(price, S, K, T, r, option_type, q, precision, max_iterations)`**:  
  Computes the implied volatility by inverting the Black-Scholes formula using Brent’s method.

### Market Data Integration

- **`get_market_data(ticker, expiration_date, strike)`**:  
  Retrieves current stock and option data using `yfinance`.

- **`validate_model(ticker, expiration_date, strikes, option_type)`**:  
  Compares model-derived prices with market prices and returns a summary in a Pandas DataFrame.

### Visualization Tools

- **`plot_option_price_surface(S, K_range, T_range, r, sigma, option_type, q)`**:  
  Generates a 3D surface plot of option prices as functions of strike price and time to expiration.

- **`plot_greeks_vs_spot(S_range, K, T, r, sigma, option_type, q)`**:  
  Plots the variation of key Greeks as the underlying stock price changes.

- **`plot_implied_volatility_smile(ticker, expiration_date)`**:  
  Plots the implied volatility smile from market data.

---

## Usage Examples

Below are some sample commands from the main block of the code:

- **Black-Scholes Pricing & Greeks:**
  ```python
  S = 200    # current stock price
  K = 180    # strike price
  T = 30/365 # 30 days to expiration
  r = 0.02   # 2% risk-free rate
  sigma = 0.15  # 15% volatility
  q = 0.0    # no dividends

  call_price = black_scholes_price(S, K, T, r, sigma, 'call', q)
  print(f"Call Price = ${call_price:.4f}")
  
  call_greeks = black_scholes_greeks(S, K, T, r, sigma, 'call', q)
  for greek, value in call_greeks.items():
      print(f"{greek.capitalize()} = {value:.6f}")

