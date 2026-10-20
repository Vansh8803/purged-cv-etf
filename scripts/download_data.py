"""Download and cache daily OHLCV for the project universe.

Run this ONCE. Every later script reads the cached CSVs in data/raw/ and
never touches the network. Yahoo restates adjusted prices after the fact,
so re-downloading would silently change results computed last week.

    python scripts/download_data.py

Re-running skips tickers that are already cached. To deliberately refresh,
pass --force, which writes a NEW file with today's date rather than
overwriting the old one.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

TICKERS = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
START = "2005-01-01"
END = "2025-12-31"

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def cache_path(ticker: str, stamp: str) -> Path:
    return RAW_DIR / f"{ticker}_{START}_{END}_dl{stamp}.csv"


def already_cached(ticker: str) -> Path | None:
    matches = sorted(RAW_DIR.glob(f"{ticker}_{START}_{END}_dl*.csv"))
    return matches[-1] if matches else None


def fetch(ticker: str) -> pd.DataFrame:
    # auto_adjust=False keeps raw OHLC *and* a separate Adj Close column.
    # We need both: the raw high/low are what the price actually touched,
    # and Adj Close is what returns should be computed from.
    df = yf.download(
        ticker,
        start=START,
        end=END,
        auto_adjust=False,
        actions=False,
        progress=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = "Date"
    return df


def describe(ticker: str, df: pd.DataFrame) -> None:
    first = df.index.min().date()
    last = df.index.max().date()
    nulls = int(df.isna().sum().sum())
    print(f"  {ticker}: {len(df):>5} rows  {first} to {last}  nulls={nulls}")
    missing = [c for c in ("Open", "High", "Low", "Close", "Adj Close", "Volume")
               if c not in df.columns]
    if missing:
        print(f"  !! {ticker}: missing columns {missing}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="download again even if a cached file exists")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    failures = []

    for ticker in TICKERS:
        existing = already_cached(ticker)
        if existing and not args.force:
            print(f"cached {ticker}: {existing.name}")
            continue

        print(f"downloading {ticker} ...")
        try:
            df = fetch(ticker)
        except Exception as exc:
            print(f"  !! {ticker} failed: {exc}")
            failures.append(ticker)
            continue

        if df.empty:
            print(f"  !! {ticker}: empty frame returned")
            failures.append(ticker)
            continue

        out = cache_path(ticker, stamp)
        df.to_csv(out)
        describe(ticker, df)
        print(f"  saved {out.relative_to(RAW_DIR.parents[1])}")

    if failures:
        print(f"\nFAILED: {', '.join(failures)}")
        return 1

    print("\nAll tickers cached. Later scripts must read data/raw/, not the network.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
