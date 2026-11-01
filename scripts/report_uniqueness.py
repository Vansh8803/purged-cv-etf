"""Average uniqueness and effective sample size, per ticker.

    python scripts/report_uniqueness.py

Reads data/processed/{ticker}_labelled.csv. Overlapping labels are not
independent observations, so the raw row count overstates how much data we
have; this prints the honest denominator beside it.

Pre-holdout rows only, and only rows whose label also RESOLVES before the
holdout starts, so nothing here touches 2023-2025 in any way.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.uniqueness import average_uniqueness, concurrency

PROCESSED = ROOT / "data" / "processed"


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    holdout_start = pd.Timestamp(cfg["split"]["holdout_start"])
    horizon = cfg["label"]["horizon_days"]

    print(f"barrier_mult={cfg['label']['barrier_mult']}  horizon={horizon} bars")
    print(f"{'tkr':>4} {'rows':>6} {'avg uniq':>9} {'eff N':>7} "
          f"{'mean conc':>10} {'max conc':>9} {'eff/raw':>8}")

    totals = []
    for ticker in cfg["universe"]:
        src = PROCESSED / f"{ticker}_labelled.csv"
        if not src.exists():
            print(f"  !! missing {src} -- run scripts/build_labels.py first")
            continue

        df = pd.read_csv(src, index_col="Date", parse_dates=True)
        df["t1"] = pd.to_datetime(df["t1"])

        # The bar axis is the full labelled calendar, so windows are never
        # truncated. Restricting it to pre-holdout bars is a date filter only.
        bars = df.index[df.index < holdout_start]
        train = df.loc[df["t1"] < holdout_start, "t1"]

        u = average_uniqueness(train, bars)
        c = concurrency(train, bars)
        eff = u.mean() * len(u)

        print(f"{ticker:>4} {len(u):>6} {u.mean():>9.4f} {eff:>7.0f} "
              f"{c.mean():>10.2f} {c.max():>9.0f} {eff / len(u):>8.3f}")
        totals.append((u.mean(), eff, len(u)))

    if totals:
        au = sum(t[0] for t in totals) / len(totals)
        mean_eff = sum(t[1] for t in totals) / len(totals)
        print(f"{'mean':>4} {'':>6} {au:>9.4f} {mean_eff:>7.0f}")
        print()
        print("Per-ticker models, so each effective N is what that one model gets.")
        print("They are not added up across tickers.")
        print(f"Concurrency cannot exceed {horizon + 1}: a window [t, t+{horizon}] spans "
              f"{horizon + 1} bars, so at most {horizon + 1} windows can cover any "
              f"one bar. The real data hits that ceiling exactly.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
