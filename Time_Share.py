"""
Enhanced End-to-End Stock Tracker & Predictor
--------------------------------------
This script demonstrates a comprehensive workflow:
  1. Data download using yfinance for selected stocks with caching.
  2. Calculation of technical indicators (SMA, EMA, RSI, MACD).
  3. Real-time updating plot with improved visualization.
  4. An optimized LSTM model with better architecture and training process.
  5. Improved derivative pricing with multiple option types.
  6. Enhanced model interpretability and evaluation metrics.
  7. Parallel processing for efficiency.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import datetime
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# PART 1: Data Acquisition & Indicator Calculation with Caching
# ---------------------------

def fetch_stock_data(ticker, period="1y", interval="1d", use_cache=True):
    """
    Download historical stock data from Yahoo Finance with caching for efficiency.
    """
    cache_dir = "stock_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{ticker}_{period}_{interval}.pkl")
    
    # Check if cached data exists and is recent (less than 1 day old)
    if use_cache and os.path.exists(cache_file):
        file_time = os.path.getmtime(cache_file)
        if (datetime.datetime.now() - datetime.datetime.fromtimestamp(file_time)).days < 1:
            try:
                with open(cache_file, 'rb') as f:
                    df = pickle.load(f)
                print(f"Loaded cached data for {ticker}")
                return df
            except Exception as e:
                print(f"Error loading cache: {e}")
    
    # Download fresh data
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            raise ValueError(f"No data found for {ticker}")
        
        df.dropna(inplace=True)
        
        # Cache the data
        if use_cache:
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)
        
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def add_technical_indicators(df):
    """
    Calculate and add several technical indicators with optimized calculations.
    """
    # Simple Moving Averages
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()

    # Exponential Moving Averages
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # MACD and Signal line
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']

    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    # Handle division by zero
    avg_loss = avg_loss.replace(0, np.finfo(float).eps)
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    std_dev = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (std_dev * 2)
    df['BB_Lower'] = df['BB_Middle'] - (std_dev * 2)
    
    # Volume indicators
    df['Volume_SMA20'] = df['Volume'].rolling(window=20).mean()
    
    # Momentum
    df['Momentum'] = df['Close'].pct_change(periods=10) * 100
    
    return df

# ---------------------------
# PART 2: Enhanced Real-Time Updating Graph
# ---------------------------
def real_time_plot(df, ticker):
    """
    Create an enhanced real-time updating chart with multiple indicators.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Price and MA data
    xdata, price_data = [], []
    sma20_data, sma50_data = [], []
    
    # Volume data
    volume_data = []
    
    # Initialize lines
    price_line, = ax1.plot([], [], 'b-', linewidth=2, label='Close Price')
    sma20_line, = ax1.plot([], [], 'r-', linewidth=1.5, label='SMA20')
    sma50_line, = ax1.plot([], [], 'g-', linewidth=1.5, label='SMA50')
    
    # Volume bars
    volume_bars = ax2.bar([], [], width=0.8, alpha=0.5, color='b')
    
    # Set titles and labels
    ax1.set_title(f"Real-Time Tracker for {ticker}", fontsize=16)
    ax1.set_ylabel("Price ($)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    ax2.set_xlabel("Trading Days", fontsize=12)
    ax2.set_ylabel("Volume", fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Function to initialize the plot
    def init():
        ax1.set_xlim(0, len(df))
        ax1.set_ylim(df['Close'].min()*0.95, df['Close'].max()*1.05)
        
        ax2.set_xlim(0, len(df))
        ax2.set_ylim(0, df['Volume'].max() * 1.1)
        
        return price_line, sma20_line, sma50_line, volume_bars

    # Update function for each frame
    def update(frame):
        xdata.append(frame)
        price_data.append(df['Close'].iloc[frame])
        
        # Update SMA data if available
        if not np.isnan(df['SMA20'].iloc[frame]):
            sma20_data.append(df['SMA20'].iloc[frame])
        else:
            sma20_data.append(None)
            
        if not np.isnan(df['SMA50'].iloc[frame]):
            sma50_data.append(df['SMA50'].iloc[frame])
        else:
            sma50_data.append(None)
        
        # Update volume data
        volume_data.append(df['Volume'].iloc[frame])
        
        # Update lines
        price_line.set_data(xdata, price_data)
        
        # Filter out None values for SMA lines
        valid_sma20 = [(x, y) for x, y in zip(xdata, sma20_data) if y is not None]
        valid_sma50 = [(x, y) for x, y in zip(xdata, sma50_data) if y is not None]
        
        if valid_sma20:
            x_sma20, y_sma20 = zip(*valid_sma20)
            sma20_line.set_data(x_sma20, y_sma20)
        
        if valid_sma50:
            x_sma50, y_sma50 = zip(*valid_sma50)
            sma50_line.set_data(x_sma50, y_sma50)
        
        # Update volume bars
        for i, bar in enumerate(volume_bars):
            if i < len(volume_data):
                bar.set_height(volume_data[i])
                bar.set_x(i)
            else:
                bar.set_height(0)
                
        return price_line, sma20_line, sma50_line, volume_bars

    ani = FuncAnimation(fig, update, frames=range(len(df)), 
                        init_func=init, interval=100, blit=True)
    plt.show()

# ---------------------------
# PART 3: Improved LSTM Model for Price Prediction
# ---------------------------
class EnhancedLSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, num_layers=2, output_size=1, dropout=0.2):
        super(EnhancedLSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism (simplified)
        self.attention = nn.Linear(hidden_size, 1)
        
        # Fully connected layers with batch normalization
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
    
    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(1)
        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))  # lstm_out shape: [batch, seq_len, hidden_size]
        
        # Apply attention (optional - can be commented out for simpler model)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        # Alternative: just use the last output
        # context = lstm_out[:, -1, :]
        
        # Apply batch normalization
        context = self.bn(context)
        
        # Fully connected layers
        out = self.fc1(context)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

def create_sequences(data, seq_length, features=1):
    """
    Convert the time series data into sequences for LSTM training.
    Supports multivariate input (multiple features).
    """
    xs = []
    ys = []
    
    for i in range(len(data) - seq_length):
        # For multivariate input, reshape accordingly
        if features == 1:
            x_seq = data[i:i+seq_length]
        else:
            x_seq = data[i:i+seq_length, :]
            
        y_seq = data[i+seq_length, 0]  # Target is always the first column (Close price)
        xs.append(x_seq)
        ys.append(y_seq)
        
    return np.array(xs), np.array(ys).reshape(-1, 1)

def prepare_data(df, features=['Close'], seq_length=60, test_size=0.2):
    """
    Prepare data for LSTM model with multiple features option.
    """
    # Select features and scale them
    data = df[features].values
    
    # Scale each feature independently
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences
    X, y = create_sequences(data_scaled, seq_length, len(features))
    
    # Split into train and test sets
    train_size = int(len(X) * (1 - test_size))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    return X_train, y_train, X_test, y_test, scaler

def train_lstm(df, features=['Close'], seq_length=60, epochs=50, lr=0.001, batch_size=32, patience=10):
    """
    Train an enhanced LSTM with early stopping and batch processing.
    """
    # Prepare data
    X_train, y_train, X_test, y_test, scaler = prepare_data(df, features, seq_length)
    
    # Create data loaders for batch processing
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize the model
    input_size = len(features)
    model = EnhancedLSTMPredictor(input_size=input_size, hidden_size=100, num_layers=2, output_size=1, dropout=0.2)
    model.to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    early_stop_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        # Training batches
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            X_test_device = X_test.to(device)
            y_test_device = y_test.to(device)
            val_outputs = model(X_test_device)
            val_loss = criterion(val_outputs, y_test_device).item()
            val_losses.append(val_loss)
        
        # Learning rate scheduler
        scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pth')
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        X_test_device = X_test.to(device)
        test_predictions = model(X_test_device).cpu().numpy()
        
    # Inverse transform predictions and actual values
    test_predictions = scaler.inverse_transform(np.concatenate([test_predictions, np.zeros((len(test_predictions), len(features)-1))], axis=1))[:, 0]
    y_test_actual = scaler.inverse_transform(np.concatenate([y_test.numpy(), np.zeros((len(y_test), len(features)-1))], axis=1))[:, 0]
    
    # Calculate metrics
    mse = mean_squared_error(y_test_actual, test_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_actual, test_predictions)
    r2 = r2_score(y_test_actual, test_predictions)
    
    print(f"\nModel Evaluation Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_actual, label='Actual Prices')
    plt.plot(test_predictions, label='Predicted Prices')
    plt.title('LSTM Model: Predictions vs Actual')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return model, scaler

def predict_future(model, scaler, recent_data, features=['Close'], seq_length=60, future_steps=5):
    """
    Predict future stock prices using the trained LSTM model.
    Supports multivariate input.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    predictions = []
    current_seq = recent_data[-seq_length:].copy()  # take last seq_length data points
    
    for _ in range(future_steps):
        # Prepare input sequence
        seq_input = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            pred = model(seq_input).cpu().numpy()[0, 0]
        
        # Prepare prediction for inverse scaling
        if len(features) > 1:
            # For multivariate, we need to create a dummy row with zeros for other features
            pred_row = np.zeros((1, len(features)))
            pred_row[0, 0] = pred  # Set the first feature (Close price)
            
            # Inverse transform
            pred_orig = scaler.inverse_transform(pred_row)[0, 0]
        else:
            # For univariate
            pred_orig = scaler.inverse_transform([[pred]])[0, 0]
        
        predictions.append(pred_orig)
        
        # Update current sequence (simulate sliding window)
        # Remove first element and add prediction at the end
        if len(features) > 1:
            # For multivariate, we need to shift all features
            # Here we're making a simplification by just shifting the first feature
            # and keeping others constant
            new_row = current_seq[-1].copy()
            new_row[0] = pred
            current_seq = np.vstack([current_seq[1:], new_row])
        else:
            # For univariate
            current_seq = np.vstack([current_seq[1:], [[pred]]])
    
    return np.array(predictions)

# ---------------------------
# PART 4: Enhanced Derivative Payoff & Model Interpretability
# ---------------------------
def derivative_payoff(predicted_price, strike_price, option_type='call', premium=0):
    """
    Calculate option payoff for different option types.
    
    Parameters:
    - predicted_price: Predicted stock price at expiration
    - strike_price: Option strike price
    - option_type: 'call', 'put', 'call_spread', 'put_spread'
    - premium: Option premium paid (for ROI calculation)
    
    Returns:
    - payoff: Option payoff at expiration
    - roi: Return on investment if premium is provided
    """
    if option_type == 'call':
        payoff = max(predicted_price - strike_price, 0)
    elif option_type == 'put':
        payoff = max(strike_price - predicted_price, 0)
    elif option_type == 'call_spread':
        # Assumes a bull call spread with strikes at strike_price and strike_price+10
        lower_payoff = max(predicted_price - strike_price, 0)
        upper_payoff = max(predicted_price - (strike_price + 10), 0)
        payoff = lower_payoff - upper_payoff
    elif option_type == 'put_spread':
        # Assumes a bear put spread with strikes at strike_price and strike_price-10
        upper_payoff = max(strike_price - predicted_price, 0)
        lower_payoff = max((strike_price - 10) - predicted_price, 0)
        payoff = upper_payoff - lower_payoff
    else:
        raise ValueError(f"Unknown option type: {option_type}")
    
    # Calculate ROI if premium is provided
    roi = None
    if premium > 0:
        roi = (payoff - premium) / premium * 100  # ROI as percentage
    
    return payoff, roi

def interpret_model(model, sample_sequence, target_idx=None):
    """
    Enhanced model interpretability using input gradients.
    This helps identify which time steps in the sequence most influence the prediction.
    
    Parameters:
    - model: Trained LSTM model
    - sample_sequence: Input sequence to analyze
    - target_idx: Optional index to analyze (default: last time step)
    
    Returns:
    - feature_importance: Normalized importance scores for each time step
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Convert to tensor and require gradients
    sample = torch.tensor(sample_sequence, dtype=torch.float32).unsqueeze(0).to(device)
    sample.requires_grad = True
    
    # Forward pass
    output = model(sample)
    
    # Backward pass (compute gradients)
    if target_idx is None:
        # Use the prediction itself as the target
        output.backward(torch.ones_like(output))
    else:
        # Create a one-hot target
        target = torch.zeros_like(output)
        target[0, target_idx] = 1
        output.backward(target)
    
    # Get gradients
    gradients = sample.grad.abs().cpu().numpy()[0]
    
    # For multivariate input, sum across features
    if len(gradients.shape) > 1 and gradients.shape[1] > 1:
        gradients = gradients.sum(axis=1)
    
    # Normalize to get relative importance
    if gradients.sum() > 0:
        feature_importance = gradients / gradients.sum()
    else:
        feature_importance = gradients
    
    return feature_importance

def visualize_feature_importance(feature_importance, df, seq_length):
    """
    Visualize which time steps in the input sequence have the most influence on the prediction.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot feature importance
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.title('Feature Importance by Time Step')
    plt.xlabel('Time Step')
    plt.ylabel('Relative Importance')
    plt.grid(True, alpha=0.3)
    
    # Add a line for the average importance
    plt.axhline(y=np.mean(feature_importance), color='r', linestyle='--', 
                label=f'Average Importance: {np.mean(feature_importance):.4f}')
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot the input sequence with importance as color intensity
    dates = df.index[-seq_length:].strftime('%Y-%m-%d')
    prices = df['Close'].values[-seq_length:]
    
    plt.figure(figsize=(14, 7))
    plt.scatter(range(len(dates)), prices, c=feature_importance, cmap='viridis', 
                s=100, alpha=0.8)
    plt.colorbar(label='Importance')
    plt.plot(range(len(dates)), prices, 'k--', alpha=0.5)
    plt.xticks(range(0, len(dates), len(dates)//10), 
               [dates[i] for i in range(0, len(dates), len(dates)//10)], 
               rotation=45)
    plt.title('Price Sequence with Feature Importance')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ---------------------------
# PART 5: Multi-Stock Analysis with Parallel Processing
# ---------------------------
def analyze_multiple_stocks(tickers, period="1y", interval="1d"):
    """
    Analyze multiple stocks in parallel for efficiency.
    """
    results = {}
    
    def process_ticker(ticker):
        print(f"Processing {ticker}...")
        df = fetch_stock_data(ticker, period, interval)
        if df is None or df.empty:
            return ticker, None
        
        df = add_technical_indicators(df)
        
        # Calculate basic statistics
        latest_close = df['Close'].iloc[-1]
        change_pct = (df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1) * 100
        avg_volume = df['Volume'].mean()
        
        # Technical signals
        sma_signal = "BUY" if df['Close'].iloc[-1] > df['SMA50'].iloc[-1] else "SELL"
        rsi = df['RSI'].iloc[-1]
        rsi_signal = "OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "NEUTRAL"
        
        macd = df['MACD'].iloc[-1]
        signal = df['Signal_Line'].iloc[-1]
        macd_signal = "BUY" if macd > signal else "SELL"
        
        return ticker, {
            'latest_close': latest_close,
            'change_pct': change_pct,
            'avg_volume': avg_volume,
            'sma_signal': sma_signal,
            'rsi': rsi,
            'rsi_signal': rsi_signal,
            'macd_signal': macd_signal,
            'dataframe': df
        }
    
    # Process tickers in parallel
    with ThreadPoolExecutor(max_workers=min(10, len(tickers))) as executor:
        for ticker, result in executor.map(process_ticker, tickers):
            if result is not None:
                results[ticker] = result
    
    # Create summary table
    summary = pd.DataFrame({
        'Ticker': list(results.keys()),
        'Latest Close': [results[t]['latest_close'] for t in results],
        'Change %': [results[t]['change_pct'] for t in results],
        'SMA Signal': [results[t]['sma_signal'] for t in results],
        'RSI': [results[t]['rsi'] for t in results],
        'RSI Signal': [results[t]['rsi_signal'] for t in results],
        'MACD Signal': [results[t]['macd_signal'] for t in results]
    })
    
    return results, summary

# ---------------------------
# MAIN EXECUTION
# ---------------------------
if __name__ == "__main__":
    # Parameters
    ticker = "AAPL"
    period = "1y"
    interval = "1d"
    
    # Fetch and process data
    print(f"Fetching data for {ticker}...")
    df = fetch_stock_data(ticker, period, interval)
    
    if df is not None:
        print("Adding technical indicators...")
        df = add_technical_indicators(df)
        print("Head of data with technical indicators:")
        print(df.head())
        
        # Real-time (simulated) updating graph for the stock
        # Uncomment the line below to view the real-time tracker.
        # real_time_plot(df, ticker)
        
        # Train enhanced LSTM model for forecasting
        print("\nTraining LSTM model...")
        seq_length = 60  # e.g., use last 60 days to predict next day
        
        # Use multiple features for better prediction
        features = ['Close', 'Volume', 'SMA20', 'RSI', 'MACD']
        
               # Filter out rows with NaN values in the selected features
        df_filtered = df.dropna(subset=features)
        
        # Train the model with more epochs and features
        model, scaler = train_lstm(df_filtered, features=features, seq_length=seq_length, 
                                  epochs=50, lr=0.001, batch_size=32, patience=10)
        
        # Predict the next 10 days of closing prices using the trained model
        data_scaled = scaler.transform(df_filtered[features].values)
        future_steps = 10
        predictions = predict_future(model, scaler, data_scaled, features=features, 
                                    seq_length=seq_length, future_steps=future_steps)
        
        print("\nFuture Predictions for the next 10 days (closing price):")
        for i, pred in enumerate(predictions):
            print(f"Day {i+1}: ${pred:.2f}")
        
        # Calculate derivative payoffs for different option types
        strike_price = df_filtered['Close'].iloc[-1]  # Use last closing price as strike
        premium = 5.0  # Example premium
        
        print(f"\nDerivative Payoffs (Strike: ${strike_price:.2f}, Premium: ${premium:.2f}):")
        for option_type in ['call', 'put', 'call_spread', 'put_spread']:
            # Use the average of predictions for simplicity
            avg_prediction = np.mean(predictions)
            payoff, roi = derivative_payoff(avg_prediction, strike_price, option_type, premium)
            print(f"{option_type.upper()}: Payoff = ${payoff:.2f}, ROI = {roi:.2f}%")
        
        # Model interpretability: Compute and visualize feature importance
        print("\nAnalyzing model interpretability...")
        sample_seq = data_scaled[-seq_length:]
        feature_importance = interpret_model(model, sample_seq)
        visualize_feature_importance(feature_importance, df_filtered, seq_length)
        
        # (Optional) Analyze multiple stocks
        print("\nAnalyzing multiple stocks...")
        other_tickers = ["MSFT", "GOOGL", "AMZN", "META"]
        results, summary = analyze_multiple_stocks([ticker] + other_tickers, period, interval)
        
        print("\nMulti-Stock Analysis Summary:")
        print(summary)
        
        # Plot the original closing price and the computed moving averages
        plt.figure(figsize=(12, 6))
        plt.plot(df['Close'], label='Close Price')
        plt.plot(df['SMA20'], label='20-Day SMA')
        plt.plot(df['SMA50'], label='50-Day SMA')
        plt.plot(df['SMA200'], label='200-Day SMA', linestyle='--')
        plt.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], alpha=0.2, color='gray', label='Bollinger Bands')
        plt.title(f"{ticker} Price and Technical Indicators")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Plot RSI
        plt.figure(figsize=(12, 4))
        plt.plot(df['RSI'], label='RSI', color='purple')
        plt.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=30, color='g', linestyle='--', alpha=0.5)
        plt.fill_between(df.index, df['RSI'], 70, where=(df['RSI'] >= 70), color='r', alpha=0.3)
        plt.fill_between(df.index, df['RSI'], 30, where=(df['RSI'] <= 30), color='g', alpha=0.3)
        plt.title(f"{ticker} RSI")
        plt.xlabel("Date")
        plt.ylabel("RSI")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Plot MACD
        plt.figure(figsize=(12, 4))
        plt.plot(df['MACD'], label='MACD', color='blue')
        plt.plot(df['Signal_Line'], label='Signal Line', color='red')
        plt.bar(df.index, df['MACD_Histogram'], label='Histogram', color='gray', alpha=0.5)
        plt.title(f"{ticker} MACD")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print(f"Failed to fetch data for {ticker}. Please check the ticker symbol and try again.")