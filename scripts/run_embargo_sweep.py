"""The embargo sensitivity sweep, 0 to 20 bars. Closes reporting item 2.

    python scripts/run_embargo_sweep.py

`project_details.md` promises the control-to-purged gap "computed for every
configuration, reported as a range rather than a single number". Until now it
existed at one embargo setting only -- whatever `embargo_frac` resolved to --
which is a single number, the thing the spec says not to report. This sweeps
every embargo from 0 to 20 bars and reports the gap as a range.

Two outputs, because one fitting pass serves both:

- `outputs/embargo_sweep.csv`: per-fold AUC for every
  (ticker, model, scheme, embargo), and the paired gap against the control.
- `outputs/positions_<ticker>.csv`: the position series the site needs at
  the site, one column per (model, scheme, embargo). Shipping positions rather
  than equity curves is the decision recorded on 2026-10-14 -- cost and
  execution lag are arithmetic the browser can do, so they do not belong in a
  precomputed cross.

The embargo is swept on the purged scheme only. The control has no embargo by
definition -- not applying one is what makes it the control -- and walk-forward
has no training row after its test block, so its embargo is a structural no-op
proven in `tests/test_walkforward.py`. Both are run once for comparison.

Pre-holdout rows only, purged against the holdout. Nothing here reads 2023-2025.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import positions
from src.cv import NaiveKFold, PurgedKFold, WalkForward
from src.evaluate import oof_predictions
from src.features import design_matrix, load_labelled, pre_holdout
from src.models import model_zoo

OUT = ROOT / "outputs"
EMBARGOS = list(range(0, 21))
ORDER = ["buy_and_hold", "ma_crossover", "logistic", "gbt",
         "meta_crossover_logistic"]


def fold_aucs(oof, y):
    """AUC within each fold, from one out-of-fold pass.

    Taken per fold rather than over the pooled predictions because the folds
    are what the standard error is computed across, and because a pooled AUC
    would silently mix five models' outputs into one ranking.
    """
    out = {}
    for fold, idx in oof.groupby("fold").groups.items():
        truth = y.loc[idx]
        if truth.nunique() > 1:
            out[int(fold)] = roc_auc_score(truth, oof.loc[idx, "proba"])
    return out


def run_scheme(model, X, y, t1, bars, cv):
    """One fitting pass; returns per-fold AUCs and the position series."""
    oof = oof_predictions(model, X, y, cv)
    pos = positions(oof["pred"] == 1, t1, bars)
    return fold_aucs(oof, y), pos


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    holdout_start = pd.Timestamp(cfg["split"]["holdout_start"])
    folds, seed = cfg["split"]["n_folds"], cfg["seed"]

    OUT.mkdir(exist_ok=True)
    rows = []

    for ticker in cfg["universe"]:
        df = load_labelled(ticker)
        bars = df.index[df.index < holdout_start]
        X, y, t1 = design_matrix(pre_holdout(df, holdout_start, df.index))
        pos_cols = {}

        for name, model in model_zoo(seed).items():
            # The two embargo-free schemes, once each.
            for scheme, cv in (("control_kfold", NaiveKFold(t1, bars, folds)),
                               ("walk_forward", WalkForward(t1, bars, folds))):
                aucs, pos = run_scheme(model, X, y, t1, bars, cv)
                for fold, auc in aucs.items():
                    rows.append({"ticker": ticker, "model": name,
                                 "scheme": scheme, "embargo": np.nan,
                                 "fold": fold, "auc": auc})
                pos_cols[f"{name}|{scheme}"] = pos.astype("int8")

            # The purged scheme, swept.
            for embargo in EMBARGOS:
                cv = PurgedKFold(t1, bars, folds, embargo)
                aucs, pos = run_scheme(model, X, y, t1, bars, cv)
                for fold, auc in aucs.items():
                    rows.append({"ticker": ticker, "model": name,
                                 "scheme": "purged_kfold", "embargo": embargo,
                                 "fold": fold, "auc": auc})
                pos_cols[f"{name}|purged_kfold|{embargo}"] = pos.astype("int8")

        pd.DataFrame(pos_cols, index=bars).to_csv(OUT / f"positions_{ticker}.csv")
        print(f"  {ticker}: swept {len(EMBARGOS)} embargos x {len(ORDER)} models")

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "embargo_sweep.csv", index=False)

    # --- The gap, paired over folds, at every embargo.
    key = ["ticker", "model", "fold"]
    control = (out[out.scheme == "control_kfold"]
               .set_index(key)["auc"].rename("control"))
    purged = out[out.scheme == "purged_kfold"].set_index(key + ["embargo"])["auc"]
    gap = (purged.rename("purged").reset_index()
           .join(control, on=key))
    gap["gap"] = gap["control"] - gap["purged"]

    print()
    print("Control minus purged AUC, by embargo. Paired over folds, then")
    print("averaged over the five tickers. The learned models are what moves;")
    print("a rule scores identically under every scheme by construction.")
    g = gap.pivot_table(index="embargo", columns="model", values="gap")[ORDER]
    print(g.round(5).to_string())

    print()
    print("The gap as a RANGE, which is what the spec asks for")
    rng = pd.DataFrame({
        "min": g.min(), "max": g.max(), "spread": g.max() - g.min(),
        "at_embargo_0": g.loc[0], "at_embargo_20": g.loc[20],
    }).loc[ORDER]
    se = (gap.groupby("model")["gap"]
          .apply(lambda s: s.std(ddof=1) / np.sqrt(len(s))).loc[ORDER])
    rng["paired_se"] = se
    print(rng.round(5).to_string())
    print("  paired_se pools every (ticker, fold, embargo) difference, so it")
    print("  is the noise the whole range has to be read against.")

    print()
    print("Purged AUC against embargo length — does withholding more hurt?")
    a = (out[out.scheme == "purged_kfold"]
         .pivot_table(index="embargo", columns="model", values="auc")[ORDER])
    print(a.loc[[0, 5, 10, 15, 20]].round(4).to_string())
    print(f"  rows dropped: embargo 0 keeps everything the purge allows;")
    print(f"  embargo 20 withholds 20 more bars after each test fold.")

    print()
    print(f"Wrote {OUT / 'embargo_sweep.csv'} and positions_<ticker>.csv")
    print("Trial counter: unchanged. A sensitivity sweep over one parameter is")
    print("a measurement of one configuration's stability, not a search for the")
    print("best setting -- we report the range rather than picking from it. If")
    print("we ever select an embargo on these numbers, the counter moves by 21.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
