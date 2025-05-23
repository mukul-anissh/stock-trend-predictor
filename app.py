import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import pandas_ta as ta
import tensorflow as tf
from datetime import date
from sklearn.preprocessing import MinMaxScaler

st.title("Stock Close Price Predictor")

ticker = st.text_input("Enter any S&P 500 ticker symbol (e.g. AAPL)", "AAPL").upper()
future_date = st.date_input(
    "Select future date to predict close price", 
    value=date.today() + pd.Timedelta(days=30)
)
if future_date <= date.today():
    st.error("Please choose a date after today.")
    st.stop()

@st.cache_resource
def load_model_and_scalers():
    model = tf.keras.models.load_model("lstm_stock_model.keras")
    scalers = pickle.load(open("scalers.pkl", "rb"))
    return model, scalers

model, scalers = load_model_and_scalers()
if ticker not in scalers:
    st.error(f"No scaler found for {ticker}. Make sure it is an S&P 500 ticker.")
    st.stop()
scaler: MinMaxScaler = scalers[ticker]

@st.cache_data(ttl=3600)
def fetch_and_preprocess(ticker: str, _scaler):
    raw = yf.download([ticker],
                      start="2020-01-01",
                      end=date.today().strftime("%Y-%m-%d"),
                      progress=False)
    df = raw.stack().reset_index()
    df.columns = ["Date", "Ticker", "Close", "High", "Low", "Open", "Volume"]

    # feature engineering to add techincal indicators
    df["RSI"] = ta.rsi(df["Close"], length=14)
    df["SMA"] = ta.sma(df["Close"], length=14)
    df["EMA"] = ta.ema(df["Close"], length=14)
    macd_df = ta.macd(close=df["Close"], fast=12, slow=26, signal=9)
    df["MACD"] = macd_df.iloc[:, 0]
    df["MACD_Signal"] = macd_df.iloc[:, 1]
    df["MACD_Hist"] = macd_df.iloc[:, 2]
    stoch_df = ta.stoch(high=df["High"], low=df["Low"], close=df["Close"], k=14, d=3)
    df["Stoch_K"] = stoch_df.iloc[:, 0]
    df["Stoch_D"] = stoch_df.iloc[:, 1]
    df["Williams %R"] = ta.willr(high=df["High"], low=df["Low"], close=df["Close"], length=14)
    df["ATR"] = ta.atr(high=df["High"], low=df["Low"], close=df["Close"], length=14)
    df["CCI"] = ta.cci(high=df["High"], low=df["Low"], close=df["Close"], length=14)
    df["ADX"] = ta.adx(high=df["High"], low=df["Low"], close=df["Close"], length=14).iloc[:, 0]  
    df["OBV"] = ta.obv(close=df["Close"], volume=df["Volume"])
    df["MFI"] = ta.mfi(high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"], length=14)

    df.dropna(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    features = ["Close","High","Low","Open","Volume", "RSI","SMA","EMA","MACD","MACD_Signal","MACD_Hist", "Stoch_K","Stoch_D","Williams %R","ATR","CCI","ADX","OBV","MFI"]
    arr = df[features].values
    arr_scaled = _scaler.transform(arr)
    df_scaled = pd.DataFrame(arr_scaled, columns=features)
    df_scaled["Date"] = df["Date"]
    df_scaled = df_scaled.set_index("Date")

    return df, df_scaled, features

df_raw, df_scaled, feature_list = fetch_and_preprocess(ticker, scaler)

rollback = 40
if len(df_scaled) < rollback:
    st.error(f"Not enough history ({len(df_scaled)} rows) to roll back {rollback} days.")
    st.stop()
X_seq = df_scaled.values[-rollback:].reshape(1, rollback, -1)

last_date = df_scaled.index[-1].date()
days_to_predict = (future_date - last_date).days
if days_to_predict < 1:
    st.error(f"Future date must be after last available date: {last_date}")
    st.stop()

preds = []
for _ in range(days_to_predict):
    scaled_close = model.predict(X_seq, verbose=0)[0, 0]
    preds.append(scaled_close)
    last_row = X_seq[0, -1, :].copy()
    ci = feature_list.index("Close")
    last_row[ci] = scaled_close
    X_seq = np.concatenate([X_seq[:, 1:, :], last_row.reshape(1, 1, -1)], axis=1)

ci = feature_list.index("Close")
dummy = np.zeros((len(preds), len(feature_list)))
dummy[:, ci] = preds
orig = scaler.inverse_transform(dummy)
preds_unscaled = orig[:, ci]
predicted_price = preds_unscaled[-1]

from datetime import timedelta

last_date = df_scaled.index[-1].date()
forecast_dates = pd.date_range(
    start=last_date + timedelta(days=1),
    end=future_date,
    freq='D'
)
pred_series = pd.Series(preds_unscaled, index=forecast_dates, name='Predicted Close')

hist = (
    df_raw[df_raw['Ticker'] == ticker]
    .set_index('Date')['Close']
    .sort_index()[-1000:]
)
full = pd.concat([hist, pred_series])

st.subheader(f"{ticker} Close Price (Historical + Forecast)")
st.line_chart(full)

predicted_price = preds_unscaled[-1]
st.write(
    f"**Predicted {ticker} close price on {future_date}:**   ",
    f"**${predicted_price:,.2f}**"
)

ticker_obj = yf.Ticker(ticker)
today_data = ticker_obj.history(period='1d')
current_price = today_data['Close'].iloc[0] if not today_data.empty else None
if current_price is not None:
    st.write(f"**Current {ticker} close price (today): ${current_price:.2f}**")
