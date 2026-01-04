# Generate sample stock data for testing
# This creates a CSV file with realistic stock price data

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("📊 Generating sample MSCI World ETF data...")

# Generate 5 years of daily data
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)
dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days only

# Simulate realistic stock price movement (geometric Brownian motion)
np.random.seed(42)  # For reproducibility
n_days = len(dates)
initial_price = 100.0
drift = 0.0003  # Daily drift (upward trend)
volatility = 0.012  # Daily volatility

# Generate price series
returns = np.random.normal(drift, volatility, n_days)
price_series = initial_price * np.exp(np.cumsum(returns))

# Create OHLCV data
data = pd.DataFrame({
    'Date': dates,
    'Open': price_series * (1 + np.random.uniform(-0.005, 0.005, n_days)),
    'High': price_series * (1 + np.random.uniform(0, 0.01, n_days)),
    'Low': price_series * (1 - np.random.uniform(0, 0.01, n_days)),
    'Close': price_series,
    'Adj Close': price_series,
    'Volume': np.random.randint(10000000, 100000000, n_days)
})

# Ensure High is highest and Low is lowest
data['High'] = data[['Open', 'High', 'Low', 'Close']].max(axis=1)
data['Low'] = data[['Open', 'High', 'Low', 'Close']].min(axis=1)

# Save to CSV
filename = 'stock_data.csv'
data.to_csv(filename, index=False)

print(f"✅ Sample data saved to '{filename}'")
print(f"📅 Date range: {data['Date'].min().date()} to {data['Date'].max().date()}")
print(f"📊 Total records: {len(data)}")
print(f"💵 Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
print(f"\nFirst 5 rows:")
print(data.head())
print(f"\nLast 5 rows:")
print(data.tail())
print(f"\n✅ You can now run: python stock_prediction_csv.py")
