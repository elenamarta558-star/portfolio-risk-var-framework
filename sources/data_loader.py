import os
import yfinance as yf
import pandas as pd
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def fetch_and_freeze_data():
    """Downloads raw prices once and saves to CSV to ensure reproducible results."""
    os.makedirs("data", exist_ok=True)
    if os.path.exists(config.RAW_DATA_PATH):
        print(f"📂 Loading frozen data from {config.RAW_DATA_PATH}")
        return pd.read_csv(config.RAW_DATA_PATH, index_col=0, parse_dates=True)
    
    print("📥 Downloading historical data from Yahoo Finance...")
    df_raw = yf.download(config.TICKERS, start=config.START_DATE, group_by='ticker')
    df = pd.DataFrame()
    for t in config.TICKERS:
        df[t] = df_raw[t]['Adj Close'] if 'Adj Close' in df_raw[t].columns else df_raw[t]['Close']
    
    df = df.dropna(thresh=4).ffill().dropna()
    df.to_csv(config.RAW_DATA_PATH)
    print(f"💾 Data frozen and saved to {config.RAW_DATA_PATH}. Total days: {len(df)}")
    return df

def get_portfolio_returns(price_df):
    """
    Computes mathematically correct portfolio log returns:
    Aggregate daily simple returns using weights, then convert the portfolio series to log returns.
    """
    # 1. Simple daily returns for assets
    asset_simple_ret = price_df.pct_change().dropna()
    
    # 2. Portfolio daily simple returns (weighted sum)
    port_simple_ret = asset_simple_ret.dot(config.WEIGHTS)
    
    # 3. Convert back to log returns for statistical analysis
    # Avoid negative values inside log by clipping near -0.99
    port_simple_ret = np.clip(port_simple_ret, -0.99, None)
    portfolio_log_returns = np.log(1 + port_simple_ret)
    asset_log_returns = np.log(1 + asset_simple_ret)
    
    return asset_log_returns, portfolio_log_returns

if __name__ == "__main__":
    df = fetch_and_freeze_data()
    a, p = get_portfolio_returns(df)
    print(f"Effective Data Start Date: {p.index[0].strftime('%Y-%m-%d')}")



