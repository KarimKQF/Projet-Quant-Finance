
import pandas as pd
import numpy as np

def calculate_sma(series, window=14):
    """Simple Moving Average."""
    return series.rolling(window=window).mean()

def calculate_rsi(series, window=14):
    """Relative Strength Index."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calculate_macd(series, fast=12, slow=26, signal=9):
    """Moving Average Convergence Divergence."""
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(series, window=20, num_std=2):
    """Bollinger Bands."""
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def calculate_atr(high, low, close, window=14):
    """Average True Range."""
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=window).mean()

def add_technical_indicators(df):
    """
    Adds advanced technical indicators to the dataframe.
    """
    df = df.copy()
    
    # 1. Moving Averages
    df['SMA_10'] = calculate_sma(df['Close'], 10)
    df['SMA_50'] = calculate_sma(df['Close'], 50)
    
    # 2. RSI
    df['RSI'] = calculate_rsi(df['Close'], 14)
    
    # 3. MACD
    df['MACD'], df['MACD_Signal'] = calculate_macd(df['Close'])
    
    # 4. Bollinger Bands
    df['BB_Upper'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
    
    # 5. ATR
    df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'])
    
    # 6. Returns (Log returns are generally better for models)
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 7. Target Variable: 1 if Next Day Close > Current Close, else 0
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Drop NaNs created by rolling windows
    df.dropna(inplace=True)
    
    return df
