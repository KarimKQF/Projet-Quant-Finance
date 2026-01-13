
import numpy as np
import pandas as pd

def generate_gbm_data(n_samples=1000, initial_price=100, mu=0.0002, sigma=0.01):
    """
    Generates synthetic OHLCV data using Geometric Brownian Motion.
    Returns a pandas DataFrame.
    """
    dt = 1 # Daily steps
    prices = [initial_price]
    
    # Generate Close prices
    for _ in range(n_samples):
        prev_price = prices[-1]
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * np.random.normal()
        price = prev_price * np.exp(drift + diffusion)
        prices.append(price)
        
    prices = prices[1:] # Remove initial price for alignment
    
    # Generate High, Low, Open based on Close
    opens = []
    highs = []
    lows = []
    volumes = []
    
    for close in prices:
        # Synthetic intray-day volatility
        daily_vol = np.random.normal(0, sigma/2)
        open_p = close * (1 + np.random.normal(0, sigma/4))
        high_p = max(open_p, close) * (1 + abs(np.random.normal(0, sigma/4)))
        low_p = min(open_p, close) * (1 - abs(np.random.normal(0, sigma/4)))
        volume = int(np.random.normal(1000000, 200000))
        
        opens.append(open_p)
        highs.append(high_p)
        lows.append(low_p)
        volumes.append(max(0, volume))
        
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='B')
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': prices,
        'Volume': volumes
    })
    
    return df.set_index('Date')
