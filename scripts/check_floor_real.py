"""The no-signal floor, measured on real bars instead of a Gaussian stand-in.

    python scripts/check_floor_real.py

`src/synthetic.py` sets NO_SIGNAL_FLOOR = 0.544 from a constant-volatility
Gaussian random walk. Real ETFs have volatility clustering and fat tails, and
the floor is produced by a volatility channel, so whether that number
transfers is a fair question rather than a pedantic one.

This answers it directly. Each ticker's own returns get a random sign flip,
which keeps the volatility path essentially intact and destroys direction, and
the whole pipeline then runs on the result -- same features, same barriers,
same purged folds, same sample size. An ordinary randomisation test.

Pre-holdout only, purged against the holdout. Nothing here reads 2023-2025.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cv import PurgedKFold
from src.data_process import engineer_features
from src.evaluate import cross_val_scores, summarise
from src.features import design_matrix, make_pipeline
from src.labels import compute_barriers, compute_labels
from src.purging import embargo_bars, purged_holdout_mask
from src.synthetic import NO_SIGNAL_FLOOR, sign_flipped_ohlc

N_SEEDS = 6


def floor_for(raw, cfg):
    lab, split = cfg["label"], cfg["split"]
    holdout_start = pd.Timestamp(split["holdout_start"])

    df = engineer_features(raw)
    df = compute_barriers(df, barrier_mult=lab["barrier_mult"],
                          vol_lookback=lab["vol_lookback_days"])
    df = df.dropna(subset=["sigma"])
    df = compute_labels(df, horizon=lab["horizon_days"])

    pre = df.loc[purged_holdout_mask(df["t1"], holdout_start, df.index)]
    X, y, t1 = design_matrix(pre)
    bars = df.index[df.index < holdout_start]

    cv = PurgedKFold(t1, bars, split["n_folds"],
                     embargo_bars(len(t1), split["embargo_frac"]))
    pipe = make_pipeline(LogisticRegression(max_iter=1000, random_state=cfg["seed"]))
    return summarise(cross_val_scores(pipe, X, y, cv))["auc"]


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    raw_dir = ROOT / cfg["data"]["raw_dir"]
    suffix = f"_{cfg['data']['start']}_{cfg['data']['end']}_dl2026-10-16.csv"

    print("No-signal floor from sign-flipped real returns")
    print(f"logistic, purged scheme, all features, {N_SEEDS} seeds per ticker\n")
    print(f"  {'ticker':>7}  {'mean':>7}{'sd':>8}{'min':>8}{'max':>8}")

    pooled, per_ticker = [], {}
    for ticker in cfg["universe"]:
        raw = pd.read_csv(raw_dir / f"{ticker}{suffix}",
                          index_col="Date", parse_dates=True)
        vals = np.array([floor_for(sign_flipped_ohlc(raw, seed=s), cfg)
                         for s in range(1, N_SEEDS + 1)])
        pooled.extend(vals)
        per_ticker[ticker] = vals.mean()
        print(f"  {ticker:>7}  {vals.mean():>7.3f}{vals.std(ddof=1):>8.3f}"
              f"{vals.min():>8.3f}{vals.max():>8.3f}")

    a = np.array(pooled)
    mean, se = a.mean(), a.std(ddof=1) / np.sqrt(len(a))
    print(f"\n  pooled: {mean:.3f} +/- {a.std(ddof=1):.3f} sd, "
          f"s.e. {se:.4f} over n={len(a)}")
    print(f"  spread across tickers: {min(per_ticker.values()):.3f} to "
          f"{max(per_ticker.values()):.3f}")
    print(f"\n  NO_SIGNAL_FLOOR currently in use: {NO_SIGNAL_FLOOR}")
    print(f"  difference: {mean - NO_SIGNAL_FLOOR:+.3f}  "
          f"({'synthetic is conservative' if mean < NO_SIGNAL_FLOOR else 'synthetic UNDERSTATES the floor'})")
    print("\n  Not wired into run_models.py. Swapping the floor changes a "
          "headline number,\n  so it is Vansh and Aksh's call -- see the "
          "2026-11-14 review entry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
