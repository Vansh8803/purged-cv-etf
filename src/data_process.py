import pandas as pd
import numpy as np
import os

def engineer_features(df):
    # 1. Price Adjustments (from your Oct 24 decision)
    adj_factor = df['Adj Close'] / df['Close']
    df['adj_high'] = df['High'] * adj_factor
    df['adj_low'] = df['Low'] * adj_factor
    
    # Base return calculation
    df['log_return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    
    # 2. Spec Features: Trailing returns (1, 5, 10, 20 days)
    for window in [1, 5, 10, 20]:
        df[f'ret_{window}d'] = df['log_return'].rolling(window=window).sum()
        
    # 3. Spec Features: Realized volatility & Vol of Vol
    df['realized_vol_20d'] = df['log_return'].rolling(window=20).std()
    df['vol_of_vol_20d'] = df['realized_vol_20d'].rolling(window=20).std()
    
    # 4. Spec Features: Distance from MA (20, 50, 200 days)
    for window in [20, 50, 200]:
        ma = df['Adj Close'].rolling(window=window).mean()
        df[f'dist_ma_{window}d'] = (df['Adj Close'] - ma) / ma
        
    # 5. Spec Features: Volume z-score (20-day rolling)
    vol_mean = df['Volume'].rolling(window=20).mean()
    vol_std = df['Volume'].rolling(window=20).std()
    df['vol_zscore_20d'] = (df['Volume'] - vol_mean) / vol_std
    
    # 6. CRITICAL SPEC RULE: Shift features one bar
    # "All computed from information available on or before day t, 
    # then shifted one bar before the model sees them."
    feature_cols = [c for c in df.columns if c not in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'adj_high', 'adj_low', 'log_return']]
    
    # We shift the features forward by 1 so that row 't' contains the features calculated up to 't-1'
    df[feature_cols] = df[feature_cols].shift(1)
    
    # Drop the massive 200-day NaN hole at the start of the dataframe caused by the 200d MA
    df = df.dropna().copy()
    
    return df

def process_all_tickers():
    tickers = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD']
    os.makedirs('data/processed', exist_ok=True)
    
    # Matching the exact download suffix from your raw data directory
    file_suffix = "_2005-01-01_2025-12-31_dl2026-10-16.csv"
    
    for ticker in tickers:
        raw_filepath = f'data/raw/{ticker}{file_suffix}'
        
        if os.path.exists(raw_filepath):
            df = pd.read_csv(raw_filepath, index_col='Date', parse_dates=True)
            df_featured = engineer_features(df)
            
            # Save to processed directory as CSV
            processed_filepath = f'data/processed/{ticker}_features.csv'
            df_featured.to_csv(processed_filepath)
            print(f"Successfully processed {ticker} -> {processed_filepath}")
        else:
            print(f"Warning: Could not find {raw_filepath}")

if __name__ == "__main__":
    process_all_tickers()