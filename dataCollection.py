from bs4 import BeautifulSoup
import requests
import yfinance as yf
import pandas as pd
import talib as ta

# using beautiful soup to scrape wikipedia for S&P 500 tickers
response = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
soup = BeautifulSoup(response.text, 'html.parser')
table = soup.find('table', {'class': 'wikitable sortable sticky-header'})
rows = table.find_all('tr')
tickers = []

for row in rows[1:]:
    ticker = row.find('a', {'class': 'external text'}).text.strip()
    tickers.append(ticker)

# fetching data from yfinance
data = yf.download(tickers, start='2020-01-01', end='2025-01-01')

# converting the multi-index DataFrame to a single index DataFrame and saving it as a csv file
df = data.stack().reset_index().drop('Adj Close', axis=1)

# feature engineering
df['RSI'] = df.groupby('Ticker')['Close'].transform(lambda x: ta.RSI(x, timeperiod=14))
df['SMA'] = df.groupby('Ticker')['Close'].transform(lambda x: ta.SMA(x, timeperiod=14))
df['EMA'] = df.groupby('Ticker')['Close'].transform(lambda x: ta.EMA(x, timeperiod=14))
df[['MACD', 'MACD_Signal', 'MACD_Hist']] = df.groupby('Ticker')['Close'].apply(lambda x: pd.DataFrame(ta.MACD(x, fastperiod=12, slowperiod=26, signalperiod=9)).T).reset_index(level=0, drop=True)
df[['Stoch_K', 'Stoch_D']] = df.groupby('Ticker').apply(lambda g: pd.DataFrame(ta.STOCHF(g['High'], g['Low'], g['Close'], fastk_period=14, fastd_period=3)).T).reset_index(level=0, drop=True)
df['Williams %R'] = df.groupby('Ticker').apply(lambda g: ta.WILLR(g['High'], g['Low'], g['Close'], timeperiod=14)).reset_index(level=0, drop=True)
df['ATR'] = df.groupby('Ticker').apply(lambda g: ta.ATR(g['High'], g['Low'], g['Close'], timeperiod=14)).reset_index(level=0, drop=True)
df['CCI'] = df.groupby('Ticker').apply(lambda g: ta.CCI(g['High'], g['Low'], g['Close'], timeperiod=14)).reset_index(level=0, drop=True)
df['ADX'] = df.groupby('Ticker').apply(lambda g: ta.ADX(g['High'], g['Low'], g['Close'], timeperiod=14)).reset_index(level=0, drop=True)
df['OBV'] = df.groupby('Ticker').apply(lambda g: ta.OBV(g['Close'], g['Volume'])).reset_index(level=0, drop=True)
df['MFI'] = df.groupby('Ticker').apply(lambda g: ta.MFI(g['High'], g['Low'], g['Close'], g['Volume'], timeperiod=14)).reset_index(level=0, drop=True)

df.dropna(inplace=True)

# converting the date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# data preprocessing
df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

scalers = {}
numerical = ['Close', 'High', 'Low', 'Open', 'Volume', 'RSI', 'SMA', 'EMA', 'MACD', 'MACD_Signal', 'MACD_Hist', 'Stoch_K', 'Stoch_D', 'Williams %R', 'ATR', 'CCI', 'ADX', 'OBV', 'MFI']
from sklearn.preprocessing import MinMaxScaler
for ticker in df['Ticker'].unique():
    scaler = MinMaxScaler()
    df.loc[df['Ticker'] == ticker, numerical] = scaler.fit_transform(df.loc[df['Ticker'] == ticker, numerical])
    scalers[ticker] = scaler

# saving the DataFrame to a csv file
df.to_csv('stocks/stocks_data.csv', index=False)

# saving the scalers to a pickle file
import pickle
with open('stocks/scalers.pkl', 'wb') as f:
    pickle.dump(scalers, f)