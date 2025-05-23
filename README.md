# stock-trend-predictor
This project implements an end-to-end pipeline for predicting future closing prices of
S&P 500 stocks using a multi‐ticker LSTM neural network and deploys it as an
interactive web app with Streamlit. We scrape tickers from Wikipedia, download
historical OHLCV data via Yahoo Finance, engineer nineteen technical indicators,
normalize per‐ticker features, and train a sequence‐to‐one LSTM to forecast the
next‐day close. Finally, we wrap the trained model in a Streamlit interface that lets users
enter any S&P 500 ticker and a target future date to visualize a 40-day historical window
plus the model’s recursive forecast and see today’s price versus the predicted future
close.

The project is live and can be accessed at https://stock-trendpredictor.streamlit.app/
