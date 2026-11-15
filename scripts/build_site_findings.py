"""Fold the headline results into docs/data.json, so the page has one source.

    python scripts/build_site_findings.py

The submission instructions say a grader must be able to understand and assess
the project from the website alone, which means the site has to carry the
results and not just the interactive panel. Those results already exist as
CSVs under `outputs/`. Hard-coding them into the HTML would mean every number
on the graded page was a transcription, and transcriptions rot.

So this reads the CSVs and writes a `findings` block into `docs/data.json`
alongside the position series. The page renders its tables from that block. Run
it after any of the result scripts, and `docs/index.html` updates with no edit.

Missing CSVs are skipped with a warning rather than failing, so the site can be
rebuilt before every experiment has been rerun.
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

from src.synthetic import NO_SIGNAL_FLOOR

OUT = ROOT / "outputs"
DOCS = ROOT / "docs"
ORDER = ["buy_and_hold", "ma_crossover", "logistic", "gbt",
         "meta_crossover_logistic"]


def rounded(df, nd=4):
    """A frame as {index: {column: value}} with NaN turned into None."""
    return json.loads(df.round(nd).to_json(orient="index"))


def read(name):
    p = OUT / name
    if not p.exists():
        print(f"  !! {name} missing — its section will be omitted")
        return None
    return pd.read_csv(p)


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    data_path = DOCS / "data.json"
    if not data_path.exists():
        print("!! docs/data.json missing — run scripts/build_site_data.py first")
        return 1

    payload = json.loads(data_path.read_text(encoding="utf-8"))
    f = {"floor_synthetic": NO_SIGNAL_FLOOR, "floor_signflip": 0.529}

    # --- the data itself
    rows = []
    for ticker in cfg["universe"]:
        lab = pd.read_csv(ROOT / "data" / "processed" / f"{ticker}_labelled.csv",
                          index_col="Date", parse_dates=True)
        pre = lab.loc[lab.index < pd.Timestamp(cfg["split"]["holdout_start"])]
        rows.append({"ticker": ticker, "rows": len(lab),
                     "pre_holdout": len(pre),
                     "label_1": round(float(pre["label"].mean()), 4),
                     "first": str(lab.index.min().date()),
                     "last": str(lab.index.max().date())})
    f["data"] = rows

    # --- step 2, from the tested module
    from src.features import load_labelled, pre_holdout
    from src.uniqueness import average_uniqueness, effective_sample_size
    uq = []
    for ticker in cfg["universe"]:
        df = load_labelled(ticker)
        hs = pd.Timestamp(cfg["split"]["holdout_start"])
        pre = pre_holdout(df, hs, df.index)
        bars = df.index[df.index < hs]
        uq.append({"ticker": ticker,
                   "rows": len(pre),
                   "uniqueness": round(float(average_uniqueness(pre["t1"], bars).mean()), 4),
                   "effective_n": int(round(effective_sample_size(pre["t1"], bars)))})
    f["uniqueness"] = uq

    # --- step 6
    s6 = read("step06_scores.csv")
    if s6 is not None:
        f["auc"] = rounded(s6.pivot_table(index="model", columns="scheme",
                                          values="auc").loc[ORDER])
        pur = s6[s6.scheme == "purged_kfold"]
        f["precision"] = rounded(pd.DataFrame({
            "precision": pur.groupby("model")["precision"].mean(),
            "coverage": pur.groupby("model")["coverage"].mean(),
            "positive_rate": pur.groupby("model")["positive_rate"].mean(),
        }).loc[ORDER])
        f["leakage"] = rounded(pd.DataFrame({
            "train_leaked": s6.groupby("scheme")["train_leak"].mean(),
            "test_touched": s6.groupby("scheme")["test_leak"].mean(),
        }))

    # --- the embargo sweep: the gap as a range
    sw = read("embargo_sweep.csv")
    if sw is not None:
        key = ["ticker", "model", "fold"]
        ctrl = sw[sw.scheme == "control_kfold"].set_index(key)["auc"].rename("c")
        pur = sw[sw.scheme == "purged_kfold"].set_index(key + ["embargo"])["auc"]
        g = pur.rename("p").reset_index().join(ctrl, on=key)
        g["gap"] = g["c"] - g["p"]
        by_emb = g.pivot_table(index="embargo", columns="model", values="gap")
        f["gap_range"] = rounded(pd.DataFrame({
            "min": by_emb.min(), "max": by_emb.max(),
            "mean": g.groupby("model")["gap"].mean(),
            "paired_se": g.groupby("model")["gap"]
                .apply(lambda s: s.std(ddof=1) / np.sqrt(len(s))),
        }).loc[ORDER], 6)

    # --- steps 7 to 9
    bt = read("step789_backtest.csv")
    if bt is not None:
        z = bt[bt.cost_bps == 0].groupby("model")
        f["strategy"] = rounded(pd.DataFrame({
            "sharpe": z["sharpe"].mean(), "sharpe_se": z["sharpe_se"].mean(),
            "sharpe_t": z["sharpe_t"].mean(), "max_drawdown": z["max_drawdown"].mean(),
            "turnover": z["turnover"].mean(), "exposure": z["exposure"].mean(),
            "matched": z["sharpe_matched"].mean(),
            "selection_edge": z["selection_edge"].mean(),
            "dsr": z["dsr"].mean(),
        }).loc[ORDER])
        f["cost_sensitivity"] = rounded(
            bt.pivot_table(index="model", columns="cost_bps",
                           values="sharpe").loc[ORDER])

    # --- the error analysis
    err = read("representative_errors.csv")
    if err is not None:
        err["date"] = pd.to_datetime(err["date"])
        f["worst_errors"] = json.loads(
            err.nsmallest(8, "realised_return")
            .assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d"))
            [["ticker", "model", "date", "confidence", "realised_return",
              "bars_held", "vol"]].round(4).to_json(orient="records"))
        err["bucket"] = pd.qcut(err["vol"], 5,
                                labels=["lowest", "low", "mid", "high", "highest"])
        f["errors_by_vol"] = rounded(
            err.pivot_table(index="bucket", columns="model",
                            values="realised_return", aggfunc="size",
                            observed=True), 0)

    # --- step 13
    hd = read("step13_holdout.csv")
    if hd is not None:
        hd["edge"] = hd["sharpe_0bps"] - hd["sharpe_matched_0bps"]
        z = hd.groupby("model")
        f["holdout"] = rounded(pd.DataFrame({
            "auc": z["auc"].mean(), "precision": z["precision"].mean(),
            "coverage": z["coverage"].mean(),
            "positive_rate": z["positive_rate"].mean(),
            "sharpe": z["sharpe_0bps"].mean(), "sharpe_se": z["sharpe_se_0bps"].mean(),
            "sharpe_t": z["sharpe_t_0bps"].mean(),
            "max_drawdown": z["max_drawdown_0bps"].mean(),
            "turnover": z["turnover_0bps"].mean(),
            "exposure": z["exposure_0bps"].mean(),
            "total_return": z["total_return_0bps"].mean(),
            "selection_edge": z["edge"].mean(), "dsr": z["dsr_0bps"].mean(),
        }).loc[ORDER])
        f["holdout_meta"] = {
            "n_train": int(hd["n_train"].mean()),
            "n_holdout": int(hd["n_holdout"].mean()),
            "bh_return_by_ticker": rounded(
                hd[hd.model == "buy_and_hold"].set_index("ticker")[["total_return_0bps"]], 3),
        }

    payload["findings"] = f
    data_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"\nWrote findings into {data_path} "
          f"({data_path.stat().st_size / 1024:,.0f} KB total)")
    print(f"  sections: {', '.join(sorted(f))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
