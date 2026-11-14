"""Step 6: score every model under every scheme, per ticker.

    python scripts/run_models.py

Reads data/processed/{ticker}_labelled.csv, writes outputs/step06_scores.csv,
and prints the comparison the write-up is built on.

Pre-holdout rows only, and *purged* against the holdout rather than merely
filtered by date -- a row dated December 2022 whose label resolves in January
2023 had its outcome decided by holdout bars. Nothing here reads 2023-2025.

Every parameter comes from config.yaml. The preprocessing is fitted inside each
fold by the pipeline, never on the whole sample.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cv import NaiveKFold, PurgedKFold, WalkForward, leakage
from src.evaluate import cross_val_scores, summarise
from src.features import design_matrix, load_labelled, pre_holdout
from src.models import model_zoo
from src.purging import embargo_bars
from src.synthetic import NO_SIGNAL_FLOOR

OUT = ROOT / "outputs"


def schemes(t1, bars, folds, embargo, seed):
    """The three ways the same model gets scored.

    Labelled exactly as in project_details.md. `control` is run deliberately
    and is wrong for this data; that is what it is for.
    """
    return {
        "control_kfold": NaiveKFold(t1, bars, folds, shuffle=False),
        "purged_kfold": PurgedKFold(t1, bars, folds, embargo),
        "walk_forward": WalkForward(t1, bars, folds, embargo),
    }


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    holdout_start = pd.Timestamp(cfg["split"]["holdout_start"])
    folds, seed = cfg["split"]["n_folds"], cfg["seed"]

    OUT.mkdir(exist_ok=True)
    rows, fold_rows = [], []

    for ticker in cfg["universe"]:
        df = load_labelled(ticker)
        # The bar axis is the full labelled calendar for this ticker. It must
        # not be the surviving rows' index: a kept row's t1 can land on a bar
        # that was itself purged, and a short calendar would raise.
        bars = df.index[df.index < holdout_start]
        pre = pre_holdout(df, holdout_start, df.index)

        X, y, t1 = design_matrix(pre)
        embargo = embargo_bars(len(t1), cfg["split"]["embargo_frac"])

        for scheme, cv in schemes(t1, bars, folds, embargo, seed).items():
            train_leak, test_leak = leakage(cv, t1)

            for name, model in model_zoo(seed).items():
                scores = cross_val_scores(model, X, y, cv)
                summary = summarise(scores)

                rows.append({"ticker": ticker, "scheme": scheme, "model": name,
                             "train_leak": train_leak, "test_leak": test_leak,
                             **summary})
                fold_rows.append(scores.assign(ticker=ticker, scheme=scheme,
                                               model=name))

        print(f"  {ticker}: {len(t1)} rows, embargo {embargo} bars")

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "step06_scores.csv", index=False)
    pd.concat(fold_rows).to_csv(OUT / "step06_folds.csv", index=False)

    order = ["buy_and_hold", "ma_crossover", "logistic", "gbt",
             "meta_crossover_logistic"]

    print("\nAUC, averaged over the five tickers")
    auc = out.pivot_table(index="model", columns="scheme", values="auc").loc[order]
    # Against the no-signal floor, not against 0.50. The label leaks through
    # volatility, so a model that knows nothing still scores ~0.544; reading
    # these against 0.50 would report the floor as a finding. The purged
    # column is the one that counts. See decision-log.md 2026-11-10.
    shown = auc.copy()
    shown["lift_vs_floor"] = auc["purged_kfold"] - NO_SIGNAL_FLOOR
    print(shown.round(4).to_string())
    print(f"  lift_vs_floor = purged AUC - {NO_SIGNAL_FLOOR}, the measured "
          f"no-signal floor. Compare it against purged_auc_se below.")

    print("\nPrecision, with the majority-class rate it has to be read against")
    prec = out.pivot_table(index="model", columns="scheme",
                           values="precision").loc[order]
    prec["majority_rate"] = out.groupby("model")["majority_rate"].mean().loc[order]
    print(prec.round(4).to_string())

    print("\nThe headline: control minus purged, in AUC")
    gap = (auc["control_kfold"] - auc["purged_kfold"]).rename("gap")
    se = out[out.scheme == "purged_kfold"].groupby("model")["auc_se"].mean().loc[order]
    print(pd.concat([gap, se.rename("purged_auc_se")], axis=1).round(4).to_string())

    print(f"\nLeakage by scheme (train rows, test rows):")
    for scheme, grp in out.groupby("scheme"):
        print(f"  {scheme:14} {100 * grp.train_leak.mean():6.2f}%  "
              f"{100 * grp.test_leak.mean():6.2f}%")

    print(f"\nWrote {OUT / 'step06_scores.csv'} and step06_folds.csv")
    print(f"Configurations scored: {len(model_zoo(seed))} models x "
          f"{len(cfg['universe'])} tickers. Schemes are measurements of the "
          f"same model, not alternatives to select between -- see the trial "
          f"counter note in decision-log.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
