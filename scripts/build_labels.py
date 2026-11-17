"""Attach barriers and triple-barrier labels to the engineered features.

Reads data/processed/{ticker}_features.csv, adds sigma, the two barrier levels,
`label` and `t1`, and writes data/processed/{ticker}_labelled.csv.

    python scripts/build_labels.py

Every parameter comes from config.yaml, so the labels and what we report
cannot drift apart.

The holdout is labelled along with everything else -- it has to be, or the rows
just before it would lose their forward window -- but the summary printed below
only ever covers pre-holdout rows. Nothing here looks at holdout outcomes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.labels import compute_barriers, compute_labels
from src.purging import purged_holdout_mask

PROCESSED = ROOT / "data" / "processed"


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    horizon = cfg["label"]["horizon_days"]
    mult = cfg["label"]["barrier_mult"]
    lookback = cfg["label"]["vol_lookback_days"]
    holdout_start = pd.Timestamp(cfg["split"]["holdout_start"])

    print(f"barrier_mult={mult}  horizon={horizon} bars  vol_lookback={lookback} days")
    print(f"{'tkr':>4} {'rows':>6} {'dropped':>8} {'label1':>7} {'timeout':>8} {'mean t1-t':>10}")

    for ticker in cfg["universe"]:
        src = PROCESSED / f"{ticker}_features.csv"
        if not src.exists():
            print(f"  !! missing {src}")
            continue

        df = pd.read_csv(src, index_col="Date", parse_dates=True)
        df = compute_barriers(df, barrier_mult=mult, vol_lookback=lookback)
        df = df.dropna(subset=["sigma"]).copy()

        before = len(df)
        out = compute_labels(df, horizon=horizon)
        out.to_csv(PROCESSED / f"{ticker}_labelled.csv")

        # Summary over pre-holdout rows only, and purged rather than merely
        # date-filtered: a December 2022 row whose label resolves in January
        # 2023 was decided by holdout bars, so even counting it here would be
        # a peek. On SPY that is 10 rows.
        pre = out.loc[purged_holdout_mask(out["t1"], holdout_start, df.index)]
        pos = {d: i for i, d in enumerate(out.index)}
        hold = pre["t1"].map(pos) - pre.index.map(pos)
        timeout = (hold == horizon).mean()
        print(f"{ticker:>4} {len(pre):>6} {before - len(out):>8} "
              f"{pre['label'].mean():>7.3f} {timeout:>8.3f} {hold.mean():>10.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
