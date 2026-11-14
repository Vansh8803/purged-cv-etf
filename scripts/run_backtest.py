"""Steps 7, 8 and 9: the strategy-level numbers.

    python scripts/run_backtest.py

Takes each model's out-of-fold predictions under the purged scheme, turns them
into a position series, costs them, and reports Sharpe with its standard error,
maximum drawdown, turnover, exposure and the Deflated Sharpe Ratio.

Writes outputs/step789_backtest.csv and outputs/step789_equity.csv.

Pre-holdout rows only, purged against the holdout. Nothing here reads 2023-2025.

**Read the Sharpe against its own standard error, not against zero.** On this
much data the interval is wide, and the whole argument of the project is that a
number without its uncertainty is not evidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import backtest, positions
from src.cv import PurgedKFold
from src.evaluate import oof_predictions
from src.features import design_matrix, load_labelled, pre_holdout
from src.metrics import exposure_matched_sharpe, summary
from src.models import model_zoo
from src.purging import embargo_bars

OUT = ROOT / "outputs"
ORDER = ["buy_and_hold", "ma_crossover", "logistic", "gbt",
         "meta_crossover_logistic"]


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    holdout_start = pd.Timestamp(cfg["split"]["holdout_start"])
    folds, seed = cfg["split"]["n_folds"], cfg["seed"]
    costs = cfg["costs_bps"]

    OUT.mkdir(exist_ok=True)
    rows, curves, net_returns = [], {}, {}

    for ticker in cfg["universe"]:
        df = load_labelled(ticker)
        bars = df.index[df.index < holdout_start]
        pre = pre_holdout(df, holdout_start, df.index)
        X, y, t1 = design_matrix(pre)

        embargo = embargo_bars(len(t1), cfg["split"]["embargo_frac"])
        cv = PurgedKFold(t1, bars, folds, embargo)
        # The trading calendar, not the surviving rows: a kept row's t1 can
        # land on a bar that was itself purged.
        log_return = df["log_return"].loc[bars]

        for name, model in model_zoo(seed).items():
            oof = oof_predictions(model, X, y, cv)
            pos = positions(oof["pred"] == 1, t1, bars)

            for cost in costs:
                bt = backtest(pos, log_return, cost_bps=cost)
                rows.append({
                    "ticker": ticker, "model": name, "cost_bps": cost,
                    **summary(bt["net_return"], traded=bt["traded"],
                              position=bt["position"]),
                })
                # Keyed by cost too: the deflation below has to use the return
                # series it is deflating, and a 10 bps strategy is a different
                # series from the same strategy gross.
                net_returns[(ticker, name, cost)] = bt["net_return"]
                if cost == costs[0]:
                    curves[(ticker, name)] = bt["equity"]
                    # Exposure-matched control: same number of bars held,
                    # chosen at random. Separates how well the model chose
                    # from how often it was invested, which a raw Sharpe
                    # against buy-and-hold confounds.
                    m, msd = exposure_matched_sharpe(
                        bt["position"], np.expm1(log_return), seed=seed)
                    rows[-1]["sharpe_matched"] = m
                    rows[-1]["sharpe_matched_sd"] = msd
                    rows[-1]["selection_edge"] = rows[-1]["sharpe"] - m

        print(f"  {ticker}: {len(t1)} rows, embargo {embargo} bars")

    out = pd.DataFrame(rows)

    # --- Step 9. The trial distribution is the spread of per-period Sharpes
    # across the configurations actually tried: five models, which is the trial
    # counter in decision-log.md. Schemes are measurements of the same model, so
    # they do not multiply it, and each ticker has its own model and its own
    # deflation. Computed per cost level, since the trials being chosen between
    # at 10 bps are the 10 bps versions.
    n_trials = len(ORDER)
    sr_std = (out.groupby(["ticker", "cost_bps"])["sharpe"]
              .std(ddof=1).div(np.sqrt(252)).to_dict())

    dsr, sr_star = [], []
    for _, r in out.iterrows():
        s = summary(net_returns[(r.ticker, r.model, r.cost_bps)],
                    n_trials=n_trials, sr_std=sr_std[(r.ticker, r.cost_bps)])
        dsr.append(s["dsr"])
        sr_star.append(s["sr_star_annual"])
    out["dsr"], out["sr_star_annual"] = dsr, sr_star

    out.to_csv(OUT / "step789_backtest.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT / "step789_equity.csv")

    show = ["sharpe", "sharpe_se", "sharpe_t", "max_drawdown", "turnover",
            "exposure"]
    print()
    print("Exposure vs selection, at 0 bps. A long-or-flat strategy invested")
    print("22% of the time forfeits most of the equity premium whatever it")
    print("picks, so Sharpe against buy-and-hold confounds two things.")
    dec = (out[out.cost_bps == costs[0]]
           .groupby("model")[["exposure", "sharpe_matched", "sharpe",
                              "selection_edge"]].mean().loc[ORDER])
    dec = dec.rename(columns={"sharpe_matched": "matched_benchmark",
                              "sharpe": "actual"})
    print(dec.round(4).to_string())
    print("  matched_benchmark = same bars held, chosen at random.")
    print("  selection_edge = actual - matched. Buy-and-hold sits at ~0 by")
    print("  construction, which is what validates the control.")
    for cost in costs:
        print(f"\nCost {cost} bps — averaged over the five tickers")
        tbl = out[out.cost_bps == cost].groupby("model")[show].mean().loc[ORDER]
        print(tbl.round(4).to_string())

    print(f"\nDeflated Sharpe, N = {n_trials} models per ticker")
    d = out.pivot_table(index="model", columns="cost_bps",
                        values="dsr").loc[ORDER]
    d.columns = [f"dsr_{c}bps" for c in d.columns]
    d.insert(0, "sr_star_annual",
             out[out.cost_bps == costs[0]].groupby("model")["sr_star_annual"]
             .mean().loc[ORDER])
    print(d.round(4).to_string())
    print("  sr_star_annual = the Sharpe the luckiest of 5 skill-less tries")
    print("  would be expected to show. DSR is the probability the observed")
    print("  Sharpe beats that rather than being the best of five coin flips.")

    print(f"\nWrote {OUT / 'step789_backtest.csv'} and step789_equity.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
