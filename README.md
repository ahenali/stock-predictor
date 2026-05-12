# Stock Price Predictor

Predicts stock closing prices using a two-layer LSTM trained on historical data pulled from Yahoo Finance. Built as a portfolio project to learn time series modelling and neural networks.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)

---

## What it does

- Pulls historical OHLCV data for any ticker via `yfinance`
- Normalises prices and builds 60-day sliding windows for the LSTM
- Trains an 80/20 train-test split, evaluates with MAE / RMSE / MAPE
- Plots actual vs predicted on the test set so you can see where the model is confident vs not
- Autoregressively forecasts N days beyond the latest data point, with a widening uncertainty band
- Everything is configurable from the sidebar — ticker, data period, forecast horizon

---

## Setup

```bash
git clone https://github.com/YOUR-USERNAME/stock-predictor.git
cd stock-predictor
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Training takes roughly 30–90 seconds depending on how much data you load and whether you have a GPU.

---

## Project structure

```
stock-predictor/
├── app.py            # everything — data, model, UI
├── requirements.txt
└── README.md
```

Deliberately kept as a single file so it's easy to read top-to-bottom and understand the full pipeline without jumping between modules.

---

## The model

```
Input → LSTM(64, return_sequences=True) → Dropout(0.2)
      → LSTM(64) → Dropout(0.2)
      → Dense(32, relu) → Dense(1)
```

Uses `EarlyStopping` on validation loss so it stops training when it stops improving rather than running all 50 epochs every time.

The 60-day window came from experimenting — shorter windows (20–30 days) were too noisy, longer ones (90+) didn't add much and slowed training.

---

## Limitations (and why that's fine)

Price-only LSTM models are a well-known starting point, not a finished product. The model doesn't know about earnings, Fed decisions, or Elon tweets. Forecasts beyond ~2 weeks drift noticeably because errors compound autoregressively.

That said, the evaluation on the held-out test set is honest — the model never sees those prices during training, so the actual vs predicted chart is a real measure of how well it generalises, not just how well it memorises.

---

## Things I'd add next

- Technical indicators as extra features (RSI, MACD, Bollinger Bands)
- Multi-step direct forecasting instead of autoregressive (less error compounding)
- Comparison against a naive baseline (just predict yesterday's price)
- Sentiment signal from news headlines via a small NLP model

---

*Not financial advice. Seriously.*
