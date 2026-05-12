import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")

# how many past days the model looks at to make each prediction
WINDOW = 60

POPULAR_TICKERS = {
    "Apple (AAPL)": "AAPL",
    "Tesla (TSLA)": "TSLA",
    "Google (GOOGL)": "GOOGL",
    "Microsoft (MSFT)": "MSFT",
    "Amazon (AMZN)": "AMZN",
    "NVIDIA (NVDA)": "NVDA",
    "Meta (META)": "META",
    "Netflix (NFLX)": "NFLX",
}


@st.cache_data(show_spinner=False)
def fetch_data(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, progress=False)
    if df.empty:
        raise ValueError(f"No data found for '{ticker}'. Check the ticker symbol.")
    return df


def make_sequences(series: np.ndarray, window: int):
    """
    Converts a 1D price series into (X, y) pairs for supervised learning.

    Example with window=3:
      prices = [10, 11, 12, 13, 14]
      X = [[10,11,12], [11,12,13]]
      y = [13, 14]

    The model learns: given these 3 days, predict the next one.
    """
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i - window:i])
        y.append(series[i])
    return np.array(X), np.array(y)


def build_model(window: int) -> Sequential:
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window, 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_and_predict(close_prices: np.ndarray):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices.reshape(-1, 1))

    # 80/20 train-test split — don't shuffle, order matters in time series
    split = int(len(scaled) * 0.8)
    train, test = scaled[:split], scaled[split:]

    X_train, y_train = make_sequences(train.flatten(), WINDOW)
    X_test, y_test   = make_sequences(
        np.concatenate([train[-WINDOW:].flatten(), test.flatten()]),
        WINDOW
    )

    # reshape for LSTM: (samples, timesteps, features)
    X_train = X_train.reshape(-1, WINDOW, 1)
    X_test  = X_test.reshape(-1, WINDOW, 1)

    model = build_model(WINDOW)
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0,
    )

    preds_scaled = model.predict(X_test, verbose=0)
    preds = scaler.inverse_transform(preds_scaled).flatten()
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    return preds, actual, split, model, scaler


def forecast_future(model, last_window: np.ndarray, scaler, days: int) -> np.ndarray:
    """Autoregressively predict `days` steps beyond the training data."""
    window = last_window.copy().tolist()
    future = []
    for _ in range(days):
        x = np.array(window[-WINDOW:]).reshape(1, WINDOW, 1)
        pred = model.predict(x, verbose=0)[0][0]
        future.append(pred)
        window.append(pred)
    return scaler.inverse_transform(np.array(future).reshape(-1, 1)).flatten()


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="wide")
st.title("📈 Stock Price Predictor")
st.markdown("Uses an LSTM neural network trained on historical price data to predict future closing prices.")
st.divider()

with st.sidebar:
    st.header("Configuration")

    mode = st.radio("Ticker input", ["Choose from list", "Enter manually"])
    if mode == "Choose from list":
        label = st.selectbox("Stock", list(POPULAR_TICKERS.keys()))
        ticker = POPULAR_TICKERS[label]
    else:
        ticker = st.text_input("Ticker symbol", value="AAPL").upper().strip()

    period = st.select_slider(
        "Training data period",
        options=["1y", "2y", "3y", "5y"],
        value="2y",
        help="More data = better model but slower training"
    )

    forecast_days = st.slider("Days to forecast ahead", 5, 60, 30)

    st.divider()
    run = st.button("Train & Predict", type="primary", use_container_width=True)

    st.caption(
        "Training takes 30–90 seconds depending on data size. "
        "The model trains fresh each run — this is intentional so you "
        "can experiment with different configs."
    )

if not run:
    st.info("👈 Configure your stock and hit **Train & Predict** to start.")
    st.markdown("""
    **How it works**

    1. Downloads historical closing prices from Yahoo Finance
    2. Normalises prices to 0–1 range (required for neural networks)
    3. Creates 60-day sliding windows as training samples
    4. Trains a 2-layer LSTM on 80% of the data
    5. Evaluates on the held-out 20% and plots actual vs predicted
    6. Autoregressively forecasts N days into the future
    """)
    st.stop()

# ── Main flow ─────────────────────────────────────────────────────────────────

with st.spinner(f"Downloading {ticker} data..."):
    try:
        df = fetch_data(ticker, period)
    except ValueError as e:
        st.error(str(e))
        st.stop()

close = df["Close"].values.flatten().astype(float)
dates = df.index

c1, c2, c3, c4 = st.columns(4)
c1.metric("Ticker", ticker)
c2.metric("Data points", f"{len(close):,}")
c3.metric("Start", str(dates[0].date()))
c4.metric("Latest close", f"${close[-1]:.2f}")

st.divider()

# raw price chart before training
with st.expander("📊 Historical price chart", expanded=False):
    fig_raw = go.Figure()
    fig_raw.add_trace(go.Scatter(
        x=dates, y=close, mode="lines",
        line=dict(color="#7F77DD", width=1.5),
        name="Close price"
    ))
    fig_raw.update_layout(
        title=f"{ticker} closing price — {period}",
        xaxis_title="Date", yaxis_title="Price (USD)",
        hovermode="x unified", height=350,
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig_raw, use_container_width=True)

with st.spinner("Training LSTM — this takes about a minute..."):
    preds, actual, split, model, scaler = train_and_predict(close)

st.success("✅ Training complete")

# evaluation metrics
mae  = np.mean(np.abs(preds - actual))
rmse = np.sqrt(np.mean((preds - actual) ** 2))
mape = np.mean(np.abs((preds - actual) / actual)) * 100

m1, m2, m3 = st.columns(3)
m1.metric("MAE",  f"${mae:.2f}",  help="Mean Absolute Error — average dollar error per prediction")
m2.metric("RMSE", f"${rmse:.2f}", help="Root Mean Squared Error — penalises large errors more")
m3.metric("MAPE", f"{mape:.2f}%", help="Mean Absolute Percentage Error — easier to interpret across stocks")

st.divider()

# actual vs predicted chart
test_dates = dates[split + WINDOW:]
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=dates[:split], y=close[:split],
    mode="lines", name="Training data",
    line=dict(color="#B4B2A9", width=1),
))
fig.add_trace(go.Scatter(
    x=test_dates, y=actual,
    mode="lines", name="Actual price",
    line=dict(color="#1D9E75", width=2),
))
fig.add_trace(go.Scatter(
    x=test_dates, y=preds,
    mode="lines", name="Predicted price",
    line=dict(color="#7F77DD", width=2, dash="dot"),
))
fig.update_layout(
    title="Actual vs predicted closing price",
    xaxis_title="Date", yaxis_title="Price (USD)",
    hovermode="x unified", height=420,
    legend=dict(orientation="h", y=-0.2),
)
st.plotly_chart(fig, use_container_width=True)

# ── Future forecast ───────────────────────────────────────────────────────────
st.subheader(f"🔮 {forecast_days}-day forecast")
st.caption(
    "The model predicts autoregressively — each prediction becomes input for the next. "
    "Errors compound over time, so treat longer forecasts as directional, not precise."
)

scaled_close = MinMaxScaler().fit_transform(close.reshape(-1, 1)).flatten()
# refit scaler on full data for forecasting
full_scaler = MinMaxScaler()
full_scaler.fit_transform(close.reshape(-1, 1))
last_window = full_scaler.transform(close.reshape(-1, 1)).flatten()[-WINDOW:]

with st.spinner("Generating forecast..."):
    future_prices = forecast_future(model, last_window, full_scaler, forecast_days)

future_dates = pd.date_range(start=dates[-1], periods=forecast_days + 1, freq="B")[1:]

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=dates[-90:], y=close[-90:],
    mode="lines", name="Recent actual",
    line=dict(color="#1D9E75", width=2),
))
fig2.add_trace(go.Scatter(
    x=future_dates, y=future_prices,
    mode="lines", name=f"Forecast ({forecast_days}d)",
    line=dict(color="#D85A30", width=2, dash="dot"),
))
# confidence band — widens over time to show growing uncertainty
upper = future_prices * (1 + np.linspace(0.01, 0.08, forecast_days))
lower = future_prices * (1 - np.linspace(0.01, 0.08, forecast_days))
fig2.add_trace(go.Scatter(
    x=np.concatenate([future_dates, future_dates[::-1]]),
    y=np.concatenate([upper, lower[::-1]]),
    fill="toself", fillcolor="rgba(216,90,48,0.1)",
    line=dict(color="rgba(255,255,255,0)"),
    name="Uncertainty band",
))
fig2.update_layout(
    title=f"{ticker} — {forecast_days}-day price forecast",
    xaxis_title="Date", yaxis_title="Price (USD)",
    hovermode="x unified", height=400,
    legend=dict(orientation="h", y=-0.2),
)
st.plotly_chart(fig2, use_container_width=True)

forecast_df = pd.DataFrame({
    "Date": future_dates.strftime("%Y-%m-%d"),
    "Forecasted price (USD)": np.round(future_prices, 2),
})
st.dataframe(forecast_df, use_container_width=True, hide_index=True)
st.download_button(
    "⬇️ Download forecast CSV",
    data=forecast_df.to_csv(index=False),
    file_name=f"{ticker}_forecast.csv",
    mime="text/csv",
)

st.divider()
st.caption(
    "⚠️ This is a portfolio project. Nothing here is financial advice. "
    "LSTMs can fit historical patterns but cannot predict the future — "
    "markets are affected by news, earnings, macro events, and other factors "
    "no price-only model can capture."
)
