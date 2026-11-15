"""Representative errors. Closes reporting item 4.

    python scripts/find_errors.py

`project_details.md` promises "representative errors -- cases where the model
said act and was wrong". A gap number contains no errors in it, so none of the
other outputs can stand in for this.

An error here is a day the model said act and the lower barrier fell first.
Under the purged scheme, on out-of-fold predictions, so every one is a day the
model had genuinely not been trained on.

Three things get reported, because a list of bad days on its own is an anecdote:

1. **How often it is wrong when it acts**, against the rate you would get by
   acting every day. That is the only honest baseline for precision.
2. **The worst individual errors**, with the date, what the model saw, and what
   the price then did. These are the cases to be able to talk about.
3. **When it fails** -- whether errors concentrate in particular volatility or
   trend conditions. This is the part that turns the list into a finding, and
   it is what the write-up can actually say something with.

Pre-holdout rows only, purged against the holdout. Nothing here reads 2023-2025.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cv import PurgedKFold
from src.evaluate import oof_predictions
from src.features import design_matrix, load_labelled, pre_holdout
from src.models import model_zoo
from src.purging import embargo_bars

OUT = ROOT / "outputs"
LEARNED = ["logistic", "gbt", "meta_crossover_logistic"]


def collect(cfg):
    """Every out-of-fold decision, with what the price actually did."""
    holdout_start = pd.Timestamp(cfg["split"]["holdout_start"])
    frames = []

    for ticker in cfg["universe"]:
        df = load_labelled(ticker)
        bars = df.index[df.index < holdout_start]
        pre = pre_holdout(df, holdout_start, df.index)
        X, y, t1 = design_matrix(pre)
        cv = PurgedKFold(t1, bars, cfg["split"]["n_folds"],
                         embargo_bars(len(t1), cfg["split"]["embargo_frac"]))

        close = df["Adj Close"]
        # What the trade actually earned: close of t to close of t1. This is
        # the trade the backtest runs, so it is the loss the error really cost.
        realised = (close.reindex(t1.to_numpy()).to_numpy()
                    / close.reindex(pre.index).to_numpy() - 1.0)
        held = (pd.DatetimeIndex(bars).searchsorted(t1.to_numpy())
                - pd.DatetimeIndex(bars).searchsorted(pre.index))

        for name in LEARNED:
            oof = oof_predictions(model_zoo(cfg["seed"])[name], X, y, cv)
            frames.append(pd.DataFrame({
                "ticker": ticker, "model": name, "date": pre.index,
                "acted": (oof["pred"] == 1).to_numpy(),
                "confidence": oof["proba"].to_numpy(),
                "label": y.to_numpy(),
                "realised_return": realised,
                "bars_held": held,
                "vol": X["realized_vol_20d"].to_numpy(),
                "trend_200d": X["dist_ma_200d"].to_numpy(),
                "ret_5d": X["ret_5d"].to_numpy(),
            }))
        print(f"  {ticker}: collected")

    return pd.concat(frames, ignore_index=True)


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    OUT.mkdir(exist_ok=True)

    d = collect(cfg)
    acts = d[d.acted]
    errors = acts[acts.label == 0].copy()
    errors.to_csv(OUT / "representative_errors.csv", index=False)

    print()
    print("1. HOW OFTEN IT IS WRONG WHEN IT ACTS")
    print("   Compared with acting every single day, which is the only")
    print("   baseline precision can be read against.")
    base = 1.0 - d.groupby("model")["label"].mean()
    tbl = pd.DataFrame({
        "acted_on": acts.groupby("model").size(),
        "of_which_wrong": errors.groupby("model").size(),
        "error_rate": 1.0 - acts.groupby("model")["label"].mean(),
        "error_rate_if_always_acting": base,
    }).loc[LEARNED]
    tbl["better_by"] = tbl.error_rate_if_always_acting - tbl.error_rate
    print(tbl.round(4).to_string())

    print()
    print("2. THE WORST ERRORS — most confident, and most expensive")
    for name in LEARNED:
        e = errors[errors.model == name]
        if e.empty:
            continue
        print(f"\n   {name} — the five costliest days it said act")
        worst = e.nsmallest(5, "realised_return")[
            ["ticker", "date", "confidence", "realised_return", "bars_held",
             "vol", "trend_200d"]]
        worst = worst.assign(date=worst["date"].dt.date)
        print(worst.to_string(index=False,
                              formatters={"confidence": "{:.3f}".format,
                                          "realised_return": "{:+.3%}".format,
                                          "vol": "{:.4f}".format,
                                          "trend_200d": "{:+.3f}".format}))

    print()
    print("3. WHEN IT FAILS — error rate by the conditions on the day")
    print("   Quintiles of each feature, computed across all acted-on days.")
    for col, label in (("vol", "trailing volatility"),
                       ("trend_200d", "distance from the 200-day average")):
        acts_q = acts.copy()
        acts_q["bucket"] = pd.qcut(acts_q[col], 5,
                                   labels=["lowest", "low", "mid", "high", "highest"])
        q = (1.0 - acts_q.pivot_table(index="bucket", columns="model",
                                      values="label", observed=True))[LEARNED]
        print(f"\n   error rate by {label}")
        print(q.round(4).to_string())

    print()
    print("4. ERROR RATE BY YEAR — are the failures spread or clustered?")
    acts_y = acts.assign(year=acts["date"].dt.year)
    y = (1.0 - acts_y.pivot_table(index="year", columns="model",
                                  values="label"))[LEARNED]
    print(y.round(3).to_string())

    print()
    print(f"Wrote {OUT / 'representative_errors.csv'}")
    print("Trial counter: unchanged. Nothing here is fitted or selected; it is")
    print("a description of predictions that already existed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
