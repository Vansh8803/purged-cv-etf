"""Open the holdout. Once.

    python scripts/open_holdout.py

The holdout is 2023-01-01 to 2025-12-31, declared in `config.yaml` on
2026-10-14 before any modelling, and untouched until now. This script is the
only thing in the repository that reads it.

**What "once" means in code.** Each model is fitted a single time, on every
pre-holdout row that survives the purge against the holdout boundary, and then
predicts the holdout. There is no cross-validation here and nothing is
selected: the models, their settings, the barrier width, the feature set and
the embargo were all fixed before this ran. Whatever comes out is the result.

**Why the training set is purged and not just date-filtered.** A row dated
December 2022 whose label resolves in January 2023 had its outcome decided by
holdout bars. Keeping it would mean training on the holdout. `purge_mask`,
through `purged_holdout_mask`, removes those -- about ten rows per ticker.

**The number is written before it is discussed.** The working agreement says
to write it down first, so this prints and saves everything in one pass and
makes no judgement about it. Reading it against the right baselines -- the
no-signal AUC floor rather than 0.50, the positive rate rather than the
majority rate, Sharpe against its own standard error -- is what the decision
log entry is for.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.metrics import precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import backtest, positions
from src.features import design_matrix, load_labelled, pre_holdout
from src.metrics import exposure_matched_sharpe, summary
from src.models import model_zoo
from src.synthetic import NO_SIGNAL_FLOOR

OUT = ROOT / "outputs"
ORDER = ["buy_and_hold", "ma_crossover", "logistic", "gbt",
         "meta_crossover_logistic"]


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    start = pd.Timestamp(cfg["split"]["holdout_start"])
    end = pd.Timestamp(cfg["split"]["holdout_end"])
    seed, costs = cfg["seed"], cfg["costs_bps"]

    OUT.mkdir(exist_ok=True)
    rows, net_returns = [], {}

    print("OPENING THE HOLDOUT")
    print(f"Declared {start.date()} to {end.date()} in config.yaml on 2026-10-14.")
    print("Each model fitted once on the purged pre-holdout rows. Nothing tuned.\n")

    for ticker in cfg["universe"]:
        df = load_labelled(ticker)

        # The bar axis comes from the RAW calendar, not from the labelled
        # frame. `compute_labels` drops the unresolvable rows at the end of the
        # series, but a kept row's t1 can still land on one of them -- those
        # are real trading days, they just never became labelled rows. Using
        # the labelled index here shortens the calendar and `positions` raises,
        # which is how this was caught rather than silently mispriced.
        raw = pd.read_csv(
            ROOT / cfg["data"]["raw_dir"]
            / f"{ticker}_{cfg['data']['start']}_{cfg['data']['end']}_dl2026-10-16.csv",
            index_col="Date", parse_dates=True)
        bars = raw.index[(raw.index >= start) & (raw.index <= end)]
        log_return = np.log(raw["Adj Close"] / raw["Adj Close"].shift(1)).loc[bars]

        train = pre_holdout(df, start, df.index)
        test = df.loc[(df.index >= start) & (df.index <= end)]
        # A holdout label must resolve inside the holdout, or the trade could
        # not have been completed within the period we are reporting on.
        test = test.loc[test["t1"] <= bars.max()]

        Xtr, ytr, _ = design_matrix(train)
        Xte, yte, t1te = design_matrix(test)

        print(f"  {ticker}: train {len(Xtr)} rows to {train.index.max().date()}, "
              f"holdout {len(Xte)} rows from {test.index.min().date()}")

        for name, model in model_zoo(seed).items():
            est = clone(model).fit(Xtr, ytr)
            proba = est.predict_proba(Xte)[:, 1]
            pred = est.predict(Xte)
            acted = pred == 1

            pos = positions(pd.Series(acted, index=Xte.index), t1te, bars)
            row = {
                "ticker": ticker, "model": name,
                "n_train": len(Xtr), "n_holdout": len(Xte),
                "auc": roc_auc_score(yte, proba) if yte.nunique() > 1 else np.nan,
                "precision": precision_score(yte, pred, zero_division=0.0),
                "coverage": float(acted.mean()),
                "positive_rate": float(yte.mean()),
            }
            for cost in costs:
                bt = backtest(pos, log_return, cost_bps=cost)
                s = summary(bt["net_return"], traded=bt["traded"],
                            position=bt["position"])
                for k in ("sharpe", "sharpe_se", "sharpe_t", "max_drawdown",
                          "turnover", "exposure", "total_return"):
                    row[f"{k}_{cost}bps"] = s[k]
                net_returns[(ticker, name, cost)] = bt["net_return"]
            m, _ = exposure_matched_sharpe(
                bt["position"], np.expm1(log_return), seed=seed)
            row["sharpe_matched_0bps"] = m
            rows.append(row)

    out = pd.DataFrame(rows)

    # Deflated Sharpe from the spread of per-period Sharpes across the five
    # models we tried, per ticker and per cost.
    sr_std = (out.groupby("ticker")["sharpe_0bps"].std(ddof=1)
              .div(np.sqrt(252)).to_dict())
    out["dsr_0bps"] = [
        summary(net_returns[(r.ticker, r.model, costs[0])],
                n_trials=len(ORDER), sr_std=sr_std[r.ticker])["dsr"]
        for _, r in out.iterrows()
    ]

    out.to_csv(OUT / "step13_holdout.csv", index=False)

    g = out.groupby("model")
    print("\n" + "=" * 72)
    print("THE HOLDOUT NUMBER — averaged over the five ETFs")
    print("=" * 72)

    print("\nClassification")
    t = pd.DataFrame({
        "auc": g["auc"].mean(),
        "lift_vs_floor": g["auc"].mean() - NO_SIGNAL_FLOOR,
        "precision": g["precision"].mean(),
        "positive_rate": g["positive_rate"].mean(),
        "coverage": g["coverage"].mean(),
    }).loc[ORDER]
    print(t.round(4).to_string())
    print(f"  floor = {NO_SIGNAL_FLOOR} (synthetic); the sign-flip null on real "
          f"bars is 0.529")
    print("  precision is read against positive_rate, not against 0.5")

    for cost in costs:
        print(f"\nStrategy at {cost} bps")
        cols = [f"sharpe_{cost}bps", f"sharpe_se_{cost}bps", f"sharpe_t_{cost}bps",
                f"max_drawdown_{cost}bps", f"turnover_{cost}bps",
                f"exposure_{cost}bps", f"total_return_{cost}bps"]
        s = g[cols].mean().loc[ORDER]
        s.columns = ["sharpe", "s.e.", "t", "max_dd", "turnover", "exposure",
                     "total_ret"]
        print(s.round(4).to_string())

    print("\nExposure-matched, and the deflated ratio, at 0 bps")
    d = pd.DataFrame({
        "sharpe": g["sharpe_0bps"].mean(),
        "matched_benchmark": g["sharpe_matched_0bps"].mean(),
        "selection_edge": g["sharpe_0bps"].mean() - g["sharpe_matched_0bps"].mean(),
        "dsr": g["dsr_0bps"].mean(),
    }).loc[ORDER]
    print(d.round(4).to_string())

    print(f"\nWrote {OUT / 'step13_holdout.csv'}")
    print("The holdout is now open. It is not to be reopened, and nothing")
    print("downstream of this point may be changed on the strength of it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
