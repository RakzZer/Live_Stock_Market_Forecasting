# Live Stock Market Forecasting
💸 Forecasting stock price of any company for next 30 days using LSTM Neural Networks

## 📊 What does this project do?

This project uses **LSTM (Long Short-Term Memory)** neural networks to predict stock prices for the next 30 days. It downloads 5 years of historical data from Yahoo Finance and trains a deep learning model to forecast future prices.

## 🚀 Features

- Downloads historical stock data from Yahoo Finance
- Trains LSTM model with 3 layers
- Predicts next 30 days of stock prices
- Generates visualization graphs
- Currently configured for **MSCI World ETF (URTH)**

## 📦 Installation

### Step 1: Create a virtual environment (recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

**Important**: If you get numpy/tensorflow compatibility errors, uninstall and reinstall:

```bash
pip uninstall numpy tensorflow keras
pip install -r requirements.txt
```

## 🎯 Usage

Simply run the Python script:

```bash
python stock_prediction.py
```

The script will:
1. Download 5 years of historical data
2. Train the LSTM model (100 epochs)
3. Generate predictions for the next 30 days
4. Save visualization plots as PNG files

## 📈 Output Files

- `model_loss.png` - Training loss over epochs
- `train_test_prediction.png` - Model performance on training/test data
- `next_30days_prediction.png` - Future 30-day prediction
- `final_prediction.png` - Complete prediction visualization

## 🔧 Configuration

To predict a different stock, edit `stock_prediction.py`:

```python
stock_symbol = 'URTH'  # Change to any Yahoo Finance ticker
```

Examples: `AAPL`, `MSFT`, `TSLA`, `^GSPC`, etc.

## 🛠️ Tech Stack

- **Python 3.x**
- **TensorFlow/Keras** - LSTM neural network
- **yfinance** - Stock market data
- **scikit-learn** - Data preprocessing
- **matplotlib** - Visualization
- **numpy/pandas** - Data manipulation

## 📚 Resources

- YouTube Video: https://youtu.be/5Gm3bWNBoWQ
- LinkedIn: https://www.linkedin.com/in/shubhambhalala/

## ⚠️ Disclaimer

This project is for educational purposes only. Stock market predictions are inherently uncertain. Do not use this for actual investment decisions without proper financial advice.
