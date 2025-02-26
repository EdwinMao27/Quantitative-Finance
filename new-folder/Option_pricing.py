"""
Enhanced Option Pricing Library
-------------------------------
This module provides comprehensive tools for option pricing and analysis:
  1. Black-Scholes analytical pricing for European options with full Greeks
  2. Monte Carlo simulation for various option types (European, American, Asian)
  3. Implied volatility calculation
  4. Binomial tree model for American options
  5. Integration with market data for model validation
  6. Visualization tools for option pricing and sensitivity analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
from scipy.optimize import brentq
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# -------------------------
# PART 1: Black-Scholes Model
# -------------------------

def black_scholes_price(S, K, T, r, sigma, option_type='call', q=0.0):
    """
    Calculate Black-Scholes price for European options.
    
    Parameters:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to expiration in years
        r (float): Risk-free interest rate (annualized)
        sigma (float): Volatility (annualized)
        option_type (str): 'call' or 'put'
        q (float): Continuous dividend yield
        
    Returns:
        float: Option price
    """
    if T <= 0:
        # At expiration, option value is its intrinsic value
        if option_type.lower() == 'call':
            return max(0, S - K)
        else:
            return max(0, K - S)
    
    # Calculate d1 and d2
    d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Standard normal CDF
    N = lambda x: stats.norm.cdf(x)
    
    if option_type.lower() == 'call':
        price = S * np.exp(-q * T) * N(d1) - K * np.exp(-r * T) * N(d2)
    else:  # Put option
        price = K * np.exp(-r * T) * N(-d2) - S * np.exp(-q * T) * N(-d1)
    
    return price

def black_scholes_greeks(S, K, T, r, sigma, option_type='call', q=0.0):
    """
    Calculate all Greeks for Black-Scholes model.
    
    Parameters:
        Same as black_scholes_price
        
    Returns:
        dict: Dictionary containing all Greeks
    """
    if T <= 0:
        # At expiration, Greeks have specific values
        if option_type.lower() == 'call':
            delta = 1.0 if S > K else 0.0
        else:
            delta = -1.0 if S < K else 0.0
        
        return {
            'delta': delta,
            'gamma': float('nan'),
            'vega': 0.0,
            'theta': float('nan'),
            'rho': 0.0,
            'charm': 0.0,
            'vanna': 0.0,
            'vomma': 0.0
        }
    
    # Calculate d1 and d2
    d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Standard normal functions
    N = lambda x: stats.norm.cdf(x)
    n = lambda x: stats.norm.pdf(x)
    
    # Common calculations
    nd1 = n(d1)
    Nd1 = N(d1)
    Nd2 = N(d2)
    
    # Calculate Greeks
    if option_type.lower() == 'call':
        delta = np.exp(-q * T) * Nd1
        theta = -((S * sigma * np.exp(-q * T)) / (2 * np.sqrt(T))) * nd1 - \
                r * K * np.exp(-r * T) * Nd2 + q * S * np.exp(-q * T) * Nd1
        rho = K * T * np.exp(-r * T) * Nd2
    else:  # Put option
        delta = np.exp(-q * T) * (Nd1 - 1)
        theta = -((S * sigma * np.exp(-q * T)) / (2 * np.sqrt(T))) * nd1 + \
                r * K * np.exp(-r * T) * (1 - Nd2) - q * S * np.exp(-q * T) * (1 - Nd1)
        rho = -K * T * np.exp(-r * T) * (1 - Nd2)
    
    # Greeks that are the same for calls and puts
    gamma = np.exp(-q * T) * nd1 / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * np.sqrt(T) * nd1
    
    # Second-order Greeks
    charm = -np.exp(-q * T) * (nd1 * (r - q) / (sigma * np.sqrt(T)) - 
                              (2 * (r - q) + sigma**2) * d2 / (2 * sigma**2 * T))
    vanna = -np.exp(-q * T) * nd1 * d2 / sigma
    vomma = vega * d1 * d2 / sigma
    
    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho,
        'charm': charm,
        'vanna': vanna,
        'vomma': vomma
    }

def implied_volatility(price, S, K, T, r, option_type='call', q=0.0, 
                       precision=0.00001, max_iterations=100):
    """
    Calculate implied volatility using bisection method.
    
    Parameters:
        price (float): Market price of the option
        Other parameters same as black_scholes_price
        precision (float): Desired precision for implied volatility
        max_iterations (int): Maximum number of iterations
        
    Returns:
        float: Implied volatility
    """
    def objective(sigma):
        return black_scholes_price(S, K, T, r, sigma, option_type, q) - price
    
    # Define reasonable bounds for volatility
    sigma_low = 0.001  # 0.1%
    sigma_high = 5.0   # 500%
    
    try:
        # Use Brent's method for root finding (more robust than bisection)
        implied_vol = brentq(objective, sigma_low, sigma_high, 
                             xtol=precision, maxiter=max_iterations)
        return implied_vol
    except ValueError:
        # If no solution is found within bounds
        if objective(sigma_low) * objective(sigma_high) > 0:
            # Check which bound gives closer price
            if abs(objective(sigma_low)) < abs(objective(sigma_high)):
                return sigma_low
            else:
                return sigma_high
        return np.nan

# -------------------------
# PART 2: Monte Carlo Simulation
# -------------------------

def monte_carlo_option_price(S, K, T, r, sigma, option_type='call', option_style='european',
                            q=0.0, n_paths=100000, n_steps=252, antithetic=True, 
                            control_variate=True, seed=None):
    """
    Enhanced Monte Carlo simulation for option pricing with variance reduction techniques.
    
    Parameters:
        S, K, T, r, sigma, option_type, q: Same as Black-Scholes
        option_style (str): 'european', 'american', 'asian', 'lookback'
        n_paths (int): Number of simulation paths
        n_steps (int): Number of time steps per path
        antithetic (bool): Use antithetic variates for variance reduction
        control_variate (bool): Use control variates for variance reduction
        seed (int): Random seed for reproducibility
        
    Returns:
        dict: Dictionary with price and error estimate
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    
    # For antithetic sampling, we'll use half the paths and generate their antithetic pairs
    actual_paths = n_paths // 2 if antithetic else n_paths
    
    # Initialize arrays
    S_paths = np.full((actual_paths, n_steps + 1), S, dtype=float)
    
    # Generate random paths
    for t in range(1, n_steps + 1):
        Z = np.random.normal(size=actual_paths)
        S_paths[:, t] = S_paths[:, t-1] * np.exp(drift + vol * Z)
    
    # If using antithetic variates, generate the antithetic paths
    if antithetic:
        S_paths_anti = np.full((actual_paths, n_steps + 1), S, dtype=float)
        for t in range(1, n_steps + 1):
            # Use negative of the random numbers used for the original paths
            Z = -np.random.normal(size=actual_paths)
            S_paths_anti[:, t] = S_paths_anti[:, t-1] * np.exp(drift + vol * Z)
        
        # Combine original and antithetic paths
        S_paths = np.vstack((S_paths, S_paths_anti))
    
    # Calculate payoffs based on option style
    if option_style.lower() == 'european':
        # European option: payoff depends only on final price
        if option_type.lower() == 'call':
            payoffs = np.maximum(S_paths[:, -1] - K, 0)
        else:  # Put
            payoffs = np.maximum(K - S_paths[:, -1], 0)
    
    elif option_style.lower() == 'asian':
        # Asian option: payoff depends on average price
        avg_prices = np.mean(S_paths[:, 1:], axis=1)  # Average along time dimension
        if option_type.lower() == 'call':
            payoffs = np.maximum(avg_prices - K, 0)
        else:  # Put
            payoffs = np.maximum(K - avg_prices, 0)
    
    elif option_style.lower() == 'lookback':
        # Lookback option: payoff depends on maximum/minimum price
        if option_type.lower() == 'call':
            payoffs = np.maximum(np.max(S_paths, axis=1) - K, 0)
        else:  # Put
            payoffs = np.maximum(K - np.min(S_paths, axis=1), 0)
    
    elif option_style.lower() == 'american':
        # American option: can be exercised at any time
        # This is a simplified approach - in practice, you'd use least squares Monte Carlo
        # or binomial trees for American options
        exercise_values = np.zeros_like(S_paths)
        if option_type.lower() == 'call':
            exercise_values = np.maximum(S_paths - K, 0)
        else:  # Put
            exercise_values = np.maximum(K - S_paths, 0)
        
        # Discount factors for each time step
        discount_factors = np.exp(-r * np.arange(n_steps + 1) * dt)
        
        # Calculate present value of exercise at each time step
        present_values = exercise_values * discount_factors.reshape(1, -1)
        
        # Optimal exercise is the maximum present value across all time steps
        payoffs = np.max(present_values, axis=1)
    
    else:
        raise ValueError(f"Unsupported option style: {option_style}")
    
    # Calculate option price (discounted expected payoff)
    option_price = np.exp(-r * T) * np.mean(payoffs)
    
    # Calculate standard error
    std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(len(payoffs))
    
    # If using control variates (for European options only)
    if control_variate and option_style.lower() == 'european':
        # Use Black-Scholes as control variate
        bs_price = black_scholes_price(S, K, T, r, sigma, option_type, q)
        
        # Calculate terminal stock prices for control variate
        if option_type.lower() == 'call':
            cv_payoffs = np.maximum(S_paths[:, -1] - K, 0)
        else:  # Put
            cv_payoffs = np.maximum(K - S_paths[:, -1], 0)
        
        cv_price = np.exp(-r * T) * np.mean(cv_payoffs)
        
        # Calculate correlation between payoffs and control variate
        covariance = np.cov(payoffs, cv_payoffs)[0, 1]
        variance_cv = np.var(cv_payoffs)
        
        # Optimal control variate coefficient
        beta = covariance / variance_cv if variance_cv > 0 else 0
        
        # Adjust option price using control variate
        option_price = option_price + beta * (bs_price - cv_price)
        
        # Adjusted standard error
        adjusted_payoffs = payoffs + beta * (bs_price - cv_payoffs)
        std_error = np.exp(-r * T) * np.std(adjusted_payoffs) / np.sqrt(len(adjusted_payoffs))
    
    return {
        'price': option_price,
        'std_error': std_error,
        'confidence_interval': (option_price - 1.96 * std_error, 
                               option_price + 1.96 * std_error)
    }

# -------------------------
# PART 3: Binomial Tree Model for American Options
# -------------------------

def binomial_tree_price(S, K, T, r, sigma, option_type='call', option_style='european',
                        q=0.0, n_steps=100):
    """
    Price options using the Cox-Ross-Rubinstein binomial tree model.
    Handles both European and American options.
    
    Parameters:
        S, K, T, r, sigma, option_type, q: Same as Black-Scholes
        option_style (str): 'european' or 'american'
        n_steps (int): Number of time steps in the tree
        
    Returns:
        float: Option price
    """
    # Time step
    dt = T / n_steps
    
    # Calculate up and down factors
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    
    # Risk-neutral probability
    p = (np.exp((r - q) * dt) - d) / (u - d)
    
    # Discount factor
    discount = np.exp(-r * dt)
    
    # Initialize asset prices at maturity (final nodes)
    prices = np.zeros(n_steps + 1)
    for i in range(n_steps + 1):
        prices[i] = S * (u ** (n_steps - i)) * (d ** i)
    
    # Initialize option values at maturity
    if option_type.lower() == 'call':
        option_values = np.maximum(prices - K, 0)
    else:  # Put
        option_values = np.maximum(K - prices, 0)
    
    # Work backwards through the tree
    for step in range(n_steps - 1, -1, -1):
        # Calculate asset prices at this step
        for i in range(step + 1):
            price = S * (u ** (step - i)) * (d ** i)
            
            # Calculate option value at this node
            option_value = discount * (p * option_values[i] + (1 - p) * option_values[i + 1])
            
            # For American options, check if early exercise is optimal
            if option_style.lower() == 'american':
                if option_type.lower() == 'call':
                    exercise_value = max(price - K, 0)
                else:  # Put
                    exercise_value = max(K - price, 0)
                
                option_value = max(option_value, exercise_value)
            
            option_values[i] = option_value
    
    # Return the option value at the root node
    return option_values[0]

# -------------------------
# PART 4: Market Data Integration
# -------------------------

def get_market_data(ticker, expiration_date=None, strike=None):
    """
    Fetch market data for a given ticker and optionally for specific options.
    
    Parameters:
        ticker (str): Stock ticker symbol
        expiration_date (str): Option expiration date in format 'YYYY-MM-DD'
        strike (float): Option strike price
        
    Returns:
        dict: Dictionary containing stock and option data
    """
    try:
        # Fetch stock data
        stock = yf.Ticker(ticker)
        current_price = stock.history(period='1d')['Close'].iloc[-1]
        
        # Get risk-free rate (using 3-month Treasury yield as proxy)
        treasury = yf.Ticker('^IRX')
        risk_free_rate = treasury.history(period='1d')['Close'].iloc[-1] / 100
        
        # Get dividend yield
        try:
            dividend_yield = stock.info.get('dividendYield', 0)
        except:
            dividend_yield = 0
        
        result = {
            'ticker': ticker,
            'current_price': current_price,
            'risk_free_rate': risk_free_rate,
            'dividend_yield': dividend_yield
        }
        
        # If expiration date is provided, fetch option chain
        if expiration_date:
            try:
                options = stock.option_chain(expiration_date)
                
                # Calculate time to expiration in years
                expiry = datetime.strptime(expiration_date, '%Y-%m-%d')
                today = datetime.now()
                T = (expiry - today).days / 365
                
                result['time_to_expiry'] = T
                
                # If strike is provided, get specific option data
                if strike:
                    # Find call option with closest strike
                    calls = options.calls
                    calls_at_strike = calls[calls['strike'] == strike]
                    
                    # Find put option with closest strike
                    puts = options.puts
                    puts_at_strike = puts[puts['strike'] == strike]
                    
                    if not calls_at_strike.empty:
                        call = calls_at_strike.iloc[0]
                        result['call'] = {
                            'strike': call['strike'],
                            'bid': call['bid'],
                            'ask': call['ask'],
                            'last_price': call['lastPrice'],
                            'volume': call['volume'],
                            'implied_volatility': call['impliedVolatility']
                        }
                    
                    if not puts_at_strike.empty:
                        put = puts_at_strike.iloc[0]
                        result['put'] = {
                            'strike': put['strike'],
                            'bid': put['bid'],
                            'ask': put['ask'],
                            'last_price': put['lastPrice'],
                            'volume': put['volume'],
                            'implied_volatility': put['impliedVolatility']
                        }
            except Exception as e:
                result['option_error'] = str(e)
        
        return result
    
    except Exception as e:
        return {'error': str(e)}

def validate_model(ticker, expiration_date, strikes=None, option_type='call'):
    """
    Validate option pricing models against market data.
    
    Parameters:
        ticker (str): Stock ticker symbol
        expiration_date (str): Option expiration date in format 'YYYY-MM-DD'
        strikes (list): List of strike prices to validate (if None, uses available strikes)
        option_type (str): 'call' or 'put'
        
    Returns:
        pd.DataFrame: Comparison of model prices vs. market prices
    """
    try:
        # Get stock data
        stock = yf.Ticker(ticker)
        options = stock.option_chain(expiration_date)
        
        # Get current stock price
        S = stock.history(period='1d')['Close'].iloc[-1]
        
        # Calculate time to expiration
        expiry = datetime.strptime(expiration_date, '%Y-%m-%d')
        today = datetime.now()
        T = (expiry - today).days / 365
        
        # Get risk-free rate (using 3-month Treasury yield as proxy)
        treasury = yf.Ticker('^IRX')
        r = treasury.history(period='1d')['Close'].iloc[-1] / 100
        
        # Get dividend yield
        try:
            q = stock.info.get('dividendYield', 0)
        except:
            q = 0
        
        # Get option chain
        if option_type.lower() == 'call':
            option_chain = options.calls
        else:
            option_chain = options.puts
        
        # Filter by strikes if provided
        if strikes:
            option_chain = option_chain[option_chain['strike'].isin(strikes)]
        
        # Prepare results
        results = []
        
        for _, option in option_chain.iterrows():
            K = option['strike']
            market_price = (option['bid'] + option['ask']) / 2 if option['bid'] > 0 and option['ask'] > 0 else option['lastPrice']
            market_iv = option['impliedVolatility']
            
            # Calculate model prices
            bs_price = black_scholes_price(S, K, T, r, market_iv, option_type, q)
            mc_result = monte_carlo_option_price(S, K, T, r, market_iv, option_type, 'european', q)
            binomial_price = binomial_tree_price(S, K, T, r, market_iv, option_type, 'european', q)
            
            results.append({
                'Strike': K,
                'Market_Price': market_price,
                'BS_Price': bs_price,
                'MC_Price': mc_result['price'],
                'Binomial_Price': binomial_price,
                'Implied_Vol': market_iv,
                'BS_Error': (bs_price - market_price) / market_price * 100,
                'MC_Error': (mc_result['price'] - market_price) / market_price * 100,
                'Binomial_Error': (binomial_price - market_price) / market_price * 100
            })
        
        return pd.DataFrame(results)
    
    except Exception as e:
        print(f"Error validating model: {e}")
        return pd.DataFrame()

# -------------------------
# PART 5: Visualization Tools
# -------------------------

def plot_option_price_surface(S, K_range, T_range, r, sigma, option_type='call', q=0.0):
    """
    Plot option price as a function of strike price and time to expiration.
    
    Parameters:
        S (float): Current stock price
        K_range (list): Range of strike prices
        T_range (list): Range of times to expiration
        r, sigma, option_type, q: Same as Black-Scholes
    """
    K_mesh, T_mesh = np.meshgrid(K_range, T_range)
    Z = np.zeros_like(K_mesh)
    
    for i in range(len(T_range)):
        for j in range(len(K_range)):
            Z[i, j] = black_scholes_price(S, K_mesh[i, j], T_mesh[i, j], r, sigma, option_type, q)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(K_mesh, T_mesh, Z, cmap='viridis', alpha=0.8)
    
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Time to Expiration (years)')
    ax.set_zlabel('Option Price')
    ax.set_title(f'{option_type.capitalize()} Option Price Surface')
    
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.show()

def plot_greeks_vs_spot(S_range, K, T, r, sigma, option_type='call', q=0.0):
    """
    Plot option Greeks as a function of the underlying price.
    
    Parameters:
        S_range (list): Range of stock prices
        K, T, r, sigma, option_type, q: Same as Black-Scholes
    """
    greeks = ['delta', 'gamma', 'theta', 'vega', 'rho']
    greek_values = {greek: [] for greek in greeks}
    
    for S in S_range:
        greek_dict = black_scholes_greeks(S, K, T, r, sigma, option_type, q)
        for greek in greeks:
            greek_values[greek].append(greek_dict[greek])
    
    fig, axes = plt.subplots(len(greeks), 1, figsize=(10, 12), sharex=True)
    
    for i, greek in enumerate(greeks):
        axes[i].plot(S_range, greek_values[greek])
        axes[i].set_ylabel(greek.capitalize())
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Stock Price')
    plt.suptitle(f'{option_type.capitalize()} Option Greeks vs. Stock Price (K={K}, T={T})')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

def plot_implied_volatility_smile(ticker, expiration_date):
    """
    Plot the implied volatility smile from market data.
    
    Parameters:
        ticker (str): Stock ticker symbol
        expiration_date (str): Option expiration date in format 'YYYY-MM-DD'
    """
    try:
        # Get stock data
        stock = yf.Ticker(ticker)
        options = stock.option_chain(expiration_date)
        
        # Get current stock price
        S = stock.history(period='1d')['Close'].iloc[-1]
        
        # Prepare data for calls and puts
        calls = options.calls
        puts = options.puts
        
        # Calculate moneyness (K/S)
        calls['moneyness'] = calls['strike'] / S
        puts['moneyness'] = puts['strike'] / S
        
        # Plot
        plt.figure(figsize=(12, 6))
        
        plt.scatter(calls['moneyness'], calls['impliedVolatility'], 
                   label='Calls', color='blue', alpha=0.7)
        plt.scatter(puts['moneyness'], puts['impliedVolatility'], 
                   label='Puts', color='red', alpha=0.7)
        
        # Add trend lines
        call_mask = (calls['volume'] > 0) & (calls['impliedVolatility'] > 0)
        put_mask = (puts['volume'] > 0) & (puts['impliedVolatility'] > 0)
        
        if sum(call_mask) > 1:
            z = np.polyfit(calls.loc[call_mask, 'moneyness'], 
                          calls.loc[call_mask, 'impliedVolatility'], 2)
            p = np.poly1d(z)
            x_range = np.linspace(min(calls['moneyness']), max(calls['moneyness']), 100)
            plt.plot(x_range, p(x_range), '--', color='blue', alpha=0.5)
        
        if sum(put_mask) > 1:
            z = np.polyfit(puts.loc[put_mask, 'moneyness'], 
                          puts.loc[put_mask, 'impliedVolatility'], 2)
            p = np.poly1d(z)
            x_range = np.linspace(min(puts['moneyness']), max(puts['moneyness']), 100)
            plt.plot(x_range, p(x_range), '--', color='red', alpha=0.5)
        
        plt.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        plt.xlabel('Moneyness (K/S)')
        plt.ylabel('Implied Volatility')
        plt.title(f'Implied Volatility Smile for {ticker} - Expiry: {expiration_date}')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    except Exception as e:
        print(f"Error plotting IV smile: {e}")

# -------------------------
# PART 6: Example Usage
# -------------------------

if __name__ == "__main__":
    # Example parameters
    S = 200    # current stock price
    K = 180    # strike price
    T = 30/365 # 30 days to expiration (in years)
    r = 0.02   # 2% risk-free rate
    sigma = 0.15  # 15% volatility
    q = 0.0    # 0% dividend yield (no dividends)
    
    # Black-Scholes pricing
    call_price = black_scholes_price(S, K, T, r, sigma, 'call', q)
    put_price = black_scholes_price(S, K, T, r, sigma, 'put', q)
    
    print("Black-Scholes Pricing:")
    print(f"Call Price = ${call_price:.4f}")
    print(f"Put Price = ${put_price:.4f}")
    
    # Calculate Greeks
    call_greeks = black_scholes_greeks(S, K, T, r, sigma, 'call', q)
    
    print("\nCall Option Greeks:")
    for greek, value in call_greeks.items():
        print(f"{greek.capitalize()} = {value:.6f}")
    
    # Monte Carlo pricing
    mc_call = monte_carlo_option_price(S, K, T, r, sigma, 'call', 'european', q)
    
        print("\nMonte Carlo Pricing:")
    print(f"Call Price = ${mc_call['price']:.4f} ± ${mc_call['std_error']:.4f}")
    print(f"Convergence ratio: {mc_call['convergence_ratio']:.6f}")
    
    # Binomial tree pricing
    binomial_call = binomial_tree_price(S, K, T, r, sigma, 'call', 'american', q)
    binomial_put = binomial_tree_price(S, K, T, r, sigma, 'put', 'american', q)
    
    print("\nBinomial Tree Pricing (American options):")
    print(f"American Call Price = ${binomial_call:.4f}")
    print(f"American Put Price = ${binomial_put:.4f}")
    
    # Implied volatility calculation
    market_price = 25.0  # Example market price
    implied_vol = implied_volatility(market_price, S, K, T, r, 'call', q)
    
    print("\nImplied Volatility:")
    print(f"Market Price = ${market_price:.2f}")
    print(f"Implied Volatility = {implied_vol:.4f} ({implied_vol*100:.2f}%)")
    
    # Sensitivity analysis
    print("\nSensitivity Analysis:")
    
    # Effect of volatility changes
    volatilities = [0.10, 0.15, 0.20, 0.25, 0.30]
    call_prices = [black_scholes_price(S, K, T, r, vol, 'call', q) for vol in volatilities]
    
    print("Effect of Volatility on Call Price:")
    for vol, price in zip(volatilities, call_prices):
        print(f"  Volatility = {vol:.2f} → Call Price = ${price:.4f}")
    
    # Effect of time to expiration
    times = [7/365, 30/365, 90/365, 180/365, 365/365]
    call_prices = [black_scholes_price(S, K, t, r, sigma, 'call', q) for t in times]
    
    print("\nEffect of Time to Expiration on Call Price:")
    for t, price in zip(times, call_prices):
        print(f"  T = {t*365:.0f} days → Call Price = ${price:.4f}")
    
    # Market data validation (uncomment to run with real market data)
    # ticker = "AAPL"
    # expiration = "2023-12-15"  # Format: YYYY-MM-DD
    # print(f"\nValidating models against market data for {ticker}, expiry {expiration}:")
    # validation = validate_model(ticker, expiration)
    # print(validation)
    
    # Visualization examples (uncomment to generate plots)
    # Plot option price surface
    # K_range = np.linspace(160, 220, 30)
    # T_range = np.linspace(0.1, 1, 30)
    # plot_option_price_surface(S, K_range, T_range, r, sigma, 'call', q)
    
    # Plot Greeks vs spot price
    # S_range = np.linspace(160, 220, 100)
    # plot_greeks_vs_spot(S_range, K, T, r, sigma, 'call', q)
    
    # Plot implied volatility smile from market data
    # plot_implied_volatility_smile("AAPL", "2023-12-15")
    
    # Demonstrate option strategy pricing
    print("\nOption Strategy Pricing:")
    
    # Bull Call Spread: Buy call at K1, sell call at K2 (K2 > K1)
    K1 = 180
    K2 = 200
    bull_spread_price = black_scholes_price(S, K1, T, r, sigma, 'call', q) - \
                        black_scholes_price(S, K2, T, r, sigma, 'call', q)
    
    print(f"Bull Call Spread (K1={K1}, K2={K2}) Price = ${bull_spread_price:.4f}")
    
    # Iron Condor: Sell put at K1, buy put at K2, sell call at K3, buy call at K4
    # where K1 < K2 < K3 < K4
    K1 = 160
    K2 = 170
    K3 = 210
    K4 = 220
    
    iron_condor_price = black_scholes_price(S, K2, T, r, sigma, 'put', q) - \
                        black_scholes_price(S, K1, T, r, sigma, 'put', q) + \
                        black_scholes_price(S, K4, T, r, sigma, 'call', q) - \
                        black_scholes_price(S, K3, T, r, sigma, 'call', q)
    
    print(f"Iron Condor Price = ${iron_condor_price:.4f}")
    
    # Demonstrate stress testing
    print("\nStress Testing:")
    
    # Stress test volatility
    base_call_price = black_scholes_price(S, K, T, r, sigma, 'call', q)
    stress_vol = sigma * 1.5  # 50% increase in volatility
    stress_call_price = black_scholes_price(S, K, T, r, stress_vol, 'call', q)
    
    print(f"Base Call Price = ${base_call_price:.4f}")
    print(f"Stressed Call Price (vol +50%) = ${stress_call_price:.4f}")
    print(f"Price Change = ${stress_call_price - base_call_price:.4f} ({(stress_call_price/base_call_price - 1)*100:.2f}%)")
    
    # Stress test underlying price
    stress_S = S * 0.9  # 10% drop in stock price
    stress_call_price = black_scholes_price(stress_S, K, T, r, sigma, 'call', q)
    
    print(f"Stressed Call Price (stock -10%) = ${stress_call_price:.4f}")
    print(f"Price Change = ${stress_call_price - base_call_price:.4f} ({(stress_call_price/base_call_price - 1)*100:.2f}%)")