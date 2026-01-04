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

### Option 1: Quick Start with Sample Data (Easiest)

Generate sample stock data and test the model:

```bash
python generate_sample_data.py
python stock_prediction_csv.py
```

This creates realistic sample data so you can see how the prediction works immediately.

### Option 2: Using Real CSV Data

If you want to use real MSCI World data but Yahoo Finance download is blocked:

1. Go to a financial data website and download historical data
2. Save it as `stock_data.csv` with columns: Date, Open, High, Low, Close, Adj Close, Volume
3. Run:
   ```bash
   python stock_prediction_csv.py
   ```

### Option 3: Auto-download from Yahoo Finance

Try auto-downloading MSCI World (URTH) data:

```bash
python stock_prediction.py
```

**Note:** If you get download errors, Yahoo Finance might be blocking requests. Use Option 1 or 2 instead.

All scripts will:
1. Load/Download 5 years of historical data
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
