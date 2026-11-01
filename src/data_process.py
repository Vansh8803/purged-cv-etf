import pandas as pd
import numpy as np
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

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

def latest_raw(ticker):
    """Newest cached download for a ticker.

    download_data.py --force writes a NEW date-stamped file rather than
    overwriting the old one, so the stamp cannot be hard-coded here.
    """
    matches = sorted((ROOT / "data" / "raw").glob(f"{ticker}_*.csv"))
    return matches[-1] if matches else None


def process_all_tickers():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    for ticker in cfg["universe"]:
        raw_filepath = latest_raw(ticker)

        if raw_filepath is None:
            print(f"Warning: no cached raw file for {ticker}")
            missing.append(ticker)
            continue

        df = pd.read_csv(raw_filepath, index_col='Date', parse_dates=True)
        df_featured = engineer_features(df)

        processed_filepath = out_dir / f'{ticker}_features.csv'
        df_featured.to_csv(processed_filepath)
        print(f"Successfully processed {ticker} -> {processed_filepath.relative_to(ROOT)}")

    if missing:
        print()
        print(f"FAILED: no raw data for {missing}. Run scripts/download_data.py.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(process_all_tickers())
