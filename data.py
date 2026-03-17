import requests
import pandas as pd
import time
from IPython.display import display
import os
import numpy as np

#**“I built a feature‑engineering pipeline that transforms raw market data into forward‑looking statistical targets (future max/min windows).

def safe_get(url, params, max_retries=10):
    for attempt in range(max_retries):
        try:
            return requests.get(url, params=params, timeout=10)
        except requests.exceptions.RequestException as e:
            wait = min(2 ** attempt, 30)  # exponential backoff, capped at 30s
            print(f"Binance is napping... retrying in {wait}s (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
    raise Exception("Binance did not wake up after all retries.")


def fetch_binance_daily_all(symbol="BTCUSDT"):
    url = "https://api.binance.com/api/v3/klines"
    interval = "1d"
    limit = 10
    all_rows = []
    start_time = 0  # beginning of time
    while True:
        params = {  "symbol": symbol, "interval": interval, "limit": limit, "startTime": start_time }
        response = safe_get(url, params=params)
        data = response.json()
        if not data:
            break  # no more data
        all_rows.extend(data)
        last_close_time = data[-1][6] # next start time = last candle close time + 1 ms
        start_time = last_close_time + 1
        time.sleep(0.2)# avoid rate limits
    df = pd.DataFrame(all_rows, columns=[ "open_time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume",
        "num_trades", "taker_buy_base", "taker_buy_quote", "ignore" ])
    # Keep only what we need
    df = df.rename(columns={"open_time": "date", "volume": "vol"})
    df["date"] = pd.to_datetime(df["date"], unit="ms").dt.date
    numeric_cols = ["open", "high", "low", "close", "vol"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    return df[["date", "open", "high", "low", "close", "vol"]]

# LOAD IF EXISTS, ELSE FETCH
DATA_FILE = "BTC_daily_data.csv"
if os.path.exists(DATA_FILE):
    print("NB: You are using saved BTCUSDT, to save time fetshing!")
    print("If you most fetch from Binance, just delete, BTC_daily_data.csv. ")
    BTC_data = pd.read_csv(DATA_FILE)
else:
    print("Fetching starting! This might take some minutes!")
    BTC_data = fetch_binance_daily_all()
    print("Fetching Finished! Your break is over lol!")
    BTC_data['hloc_avg'] = BTC_data[['high','low','open','close']].mean(axis=1)
    BTC_data['log_return'] = np.log(BTC_data['hloc_avg'])
    BTC_data.to_csv(DATA_FILE, index=False)

def rolling_ols_slope(series, window):
    slopes = []
    for i in range(len(series)):
        if i < window:
            slopes.append(np.nan)
        else:
            y = series.iloc[i-window+1:i+1].values
            x = np.arange(window)
            beta, _ = np.polyfit(x, y, 1)  # slope, intercept
            slopes.append(beta)
    return pd.Series(slopes, index=series.index)
#NB the periode of 350, 128 and 14 i based on a frequence analysis of the BTCUSDT!
BTC_data['ols_slope350'] =rolling_ols_slope(BTC_data['log_return'], 350)
BTC_data['ols_slope128'] =rolling_ols_slope(BTC_data['log_return'], 128)
BTC_data['ols_slope14']  = rolling_ols_slope(BTC_data['log_return'], 14)
# --- Targets: 3weeks gain/loss buckets ---
h = 21
future_max = BTC_data['hloc_avg'].shift(-1).rolling(h).max()
future_min = BTC_data['hloc_avg'].shift(-1).rolling(h).min()