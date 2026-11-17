"""Step 11: the precomputed grid the site reads.

    python scripts/build_site_data.py

Writes `docs/data.json`.

**What is precomputed and what is not.** All four controls crossed is
2 x 21 x 21 x 2 = 1,764 combinations, and a full equity curve for each is
hundreds of megabytes. That was decided against on 2026-10-14. Instead this
ships the *position series* -- one per purging/embargo combination -- plus each
ticker's daily returns, and the browser multiplies them together. Cost and
execution lag are then arithmetic on top, so two of the four controls need no
precomputation at all.

**Run-length encoding.** A position series is 4,312 zeros and ones, but it is
blocky: a position is held for about six bars, so a series has only a few
hundred runs. Storing run lengths instead of bars is about six times smaller
and trivial to decode. The whole grid comes to a few hundred kilobytes, small
enough to inline into one self-contained HTML file -- which matters more than
elegance for something that has to work during a viva with no server running.

**Execution lag is a one-bar delay, not a next-open fill.** The spec says
"same-bar or next-open". Doing next-open honestly needs open-to-close returns
threaded through the entry bar, which cannot be done as arithmetic on a
position series. A one-bar delay is the standard vectorised treatment of
execution lag and it *is* a real cost of being late; it is not the same thing
as an open fill, and the site says so rather than overclaiming.

Pre-holdout rows only. Nothing here reads 2023-2025.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import load_labelled

OUT = ROOT / "docs"
MODELS = ["buy_and_hold", "ma_crossover", "logistic", "gbt",
          "meta_crossover_logistic"]


def rle(bits):
    """[0,0,1,1,1,0] -> [2,3,1], always starting with a run of zeros.

    A leading zero-length run is emitted when the series starts held, so the
    decoder never needs to know which value a run represents: runs alternate,
    starting from zero.
    """
    b = np.asarray(bits, dtype=np.int8)
    change = np.flatnonzero(np.diff(b)) + 1
    bounds = np.concatenate([[0], change, [len(b)]])
    runs = np.diff(bounds).tolist()
    return runs if b[0] == 0 else [0] + runs


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    holdout_start = pd.Timestamp(cfg["split"]["holdout_start"])
    OUT.mkdir(exist_ok=True)

    payload = {
        "meta": {
            "built": pd.Timestamp.today().date().isoformat(),
            "universe": cfg["universe"],
            "models": MODELS,
            "embargos": list(range(0, 21)),
            "barrier_mult": cfg["label"]["barrier_mult"],
            "horizon_days": cfg["label"]["horizon_days"],
            "n_folds": cfg["split"]["n_folds"],
            "holdout_start": str(holdout_start.date()),
            "note": ("Pre-holdout only. Positions are run-length encoded, "
                     "runs alternating from zero. Equity, cost and execution "
                     "lag are computed in the browser."),
        },
        # One shared date axis. All five ETFs are on the same NYSE calendar and
        # the labelled frames come out the same length, so storing it per
        # ticker wasted about 220 KB of the file on four exact copies.
        "dates": None,
        "tickers": {},
    }

    for ticker in cfg["universe"]:
        pos = pd.read_csv(ROOT / "outputs" / f"positions_{ticker}.csv",
                          index_col=0, parse_dates=True)
        df = load_labelled(ticker)
        simple = np.expm1(df["log_return"].reindex(pos.index).fillna(0.0))

        series = {}
        for col in pos.columns:
            # "model|control_kfold", "model|walk_forward",
            # "model|purged_kfold|<embargo>"  ->  flat keys for the browser
            parts = col.split("|")
            key = parts[1] if len(parts) == 2 else f"purged|{parts[2]}"
            series.setdefault(parts[0], {})[key] = rle(pos[col].to_numpy())

        dates = [d.strftime("%Y-%m-%d") for d in pos.index]
        if payload["dates"] is None:
            payload["dates"] = dates
        elif dates != payload["dates"]:
            raise ValueError(f"{ticker} is not on the same bar axis as the "
                             f"others; the site assumes one shared calendar")

        payload["tickers"][ticker] = {
            # 6 dp is well inside the precision any chart or statistic needs,
            # and it roughly halves the file against full float repr.
            "returns": [round(float(x), 6) for x in simple.to_numpy()],
            "positions": series,
        }
        print(f"  {ticker}: {len(pos)} bars, {len(pos.columns)} series")

    path = OUT / "data.json"
    # Keep any findings already in the file. This script rebuilds the position
    # series; scripts/build_site_findings.py writes the results tables beside
    # them. Without this, rebuilding one wipes the other and every table on the
    # site goes blank until the second script happens to be run again.
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
        if "findings" in existing:
            payload["findings"] = existing["findings"]
            print(f"  kept {len(existing['findings'])} existing findings sections")

    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    runs = [len(r) for t in payload["tickers"].values()
            for m in t["positions"].values() for r in m.values()]
    bars = len(payload["dates"])
    print()
    print(f"Wrote {path}  ({path.stat().st_size / 1024:,.0f} KB)")
    print(f"  {len(runs)} position series, {bars} bars each.")
    print(f"  Run-length encoding stores a median of {sorted(runs)[len(runs) // 2]} "
          f"runs instead of {bars} bars, so the position data is about "
          f"{bars / (sorted(runs)[len(runs) // 2]):.0f}x smaller than bar by bar.")
    print(f"  Precomputing all 2 x 21 x 21 x 2 = 1,764 control combinations as "
          f"equity curves instead\n  would be hundreds of MB, which is what the "
          f"2026-10-14 decision avoided.")
    print(f"  Last bar {payload['dates'][-1]}; the holdout opens "
          f"{payload['meta']['holdout_start']} and is not in this file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
