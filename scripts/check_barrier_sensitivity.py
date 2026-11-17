"""How much does our headline depend on the barrier width we picked?

    python scripts/check_barrier_sensitivity.py

We fixed the barrier width at 2.5 sigma on 2026-11-01, for a reason that had
nothing to do with accuracy: at narrower widths the median label resolved in a
single bar, label windows barely overlapped, and the comparison this whole
project is built on would have had nothing to measure. This script asks what
that choice cost or gained us in predictive terms.

**Raw AUC is not comparable across widths**, which is the whole reason this
needs care. The no-signal floor moves with the width. Wide barriers time out
more often, a timeout counts as label 0, and trailing volatility predicts
timeouts — so a model with no skill at all scores higher when the barriers are
wider. Comparing AUC at 1.0 against AUC at 2.5 would mostly be comparing two
different floors.

So for every width we measure both: the real AUC, and the floor at that same
width from the sign-flip null, where each daily return keeps its magnitude and
gets a random sign. The quantity that means anything is the gap between them.

**Nothing here may change the barrier width.** The holdout was opened on
2026-11-15, and our own rule is that nothing upstream of that may move on the
strength of anything measured afterwards. This is a sensitivity analysis for
the limitations section, not a search for a better setting, which is also why
it does not touch the trial counter.

Pre-holdout rows only. This does not read 2023-2025.
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
from src.synthetic import sign_flipped_ohlc
from src.uniqueness import average_uniqueness, effective_sample_size

OUT = ROOT / "outputs"
WIDTHS = [1.0, 1.5, 2.0, 2.5, 3.0]
FLOOR_SEEDS = (1, 2, 3)


def score(feat, mult, cfg, with_stats=False):
    """Purged-CV logistic AUC on an already-featured frame at this width."""
    holdout_start = pd.Timestamp(cfg["split"]["holdout_start"])
    df = compute_barriers(feat, barrier_mult=mult,
                          vol_lookback=cfg["label"]["vol_lookback_days"])
    df = df.dropna(subset=["sigma"])
    df = compute_labels(df, horizon=cfg["label"]["horizon_days"])

    pre = df.loc[purged_holdout_mask(df["t1"], holdout_start, df.index)]
    X, y, t1 = design_matrix(pre)
    bars = df.index[df.index < holdout_start]
    cv = PurgedKFold(t1, bars, cfg["split"]["n_folds"],
                     embargo_bars(len(t1), cfg["split"]["embargo_frac"]))
    pipe = make_pipeline(LogisticRegression(max_iter=1000,
                                            random_state=cfg["seed"]))
    auc = summarise(cross_val_scores(pipe, X, y, cv))["auc"]
    if not with_stats:
        return auc, None

    pos = {d: i for i, d in enumerate(bars)}
    hold = np.array([pos[v] - pos[t] for t, v in pre["t1"].items()
                     if v in pos and t in pos], dtype=float)
    u = average_uniqueness(pre["t1"], bars)
    return auc, {
        "label1": float(pre["label"].mean()),
        "timeout": float((hold == cfg["label"]["horizon_days"]).mean()),
        "window": float(hold.mean()),
        "uniqueness": float(u.mean()),
        "effN": effective_sample_size(pre["t1"], bars),
    }


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    raw_dir = ROOT / cfg["data"]["raw_dir"]
    suffix = f"_{cfg['data']['start']}_{cfg['data']['end']}_dl2026-10-16.csv"
    OUT.mkdir(exist_ok=True)

    raw, feat = {}, {}
    for ticker in cfg["universe"]:
        raw[ticker] = pd.read_csv(raw_dir / f"{ticker}{suffix}",
                                  index_col="Date", parse_dates=True)
        feat[ticker] = engineer_features(raw[ticker].copy())

    print(f"Barrier width sensitivity. Logistic, purged CV, pre-holdout.")
    print(f"Width in use: {cfg['label']['barrier_mult']} sigma.\n")
    print(f"{'mult':>5}{'label1':>8}{'timeout':>9}{'window':>8}{'effN':>7}"
          f"{'AUC':>8}{'floor':>8}{'lift':>9}{'lift/sd':>9}")

    rows = []
    for mult in WIDTHS:
        scored = [score(feat[t].copy(), mult, cfg, with_stats=True)
                  for t in cfg["universe"]]
        aucs = [a for a, _ in scored]
        stats = [s for _, s in scored]

        floors = [score(engineer_features(sign_flipped_ohlc(raw[t], seed=sd)),
                        mult, cfg)[0]
                  for t in cfg["universe"] for sd in FLOOR_SEEDS]

        auc, floor = float(np.mean(aucs)), float(np.mean(floors))
        fsd = float(np.std(floors, ddof=1))
        m = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
        rows.append({"mult": mult, "auc": auc, "floor": floor,
                     "floor_sd": fsd, "lift": auc - floor, **m})
        print(f"{mult:>5.1f}{m['label1']:>8.3f}{m['timeout']:>9.3f}"
              f"{m['window']:>8.2f}{m['effN']:>7.0f}{auc:>8.4f}{floor:>8.4f}"
              f"{auc - floor:>+9.4f}{(auc - floor) / fsd:>9.2f}")

    pd.DataFrame(rows).to_csv(OUT / "barrier_width_sensitivity.csv", index=False)

    print("\n  lift   = AUC minus the no-signal floor at the SAME width.")
    print("  lift/sd = that lift against the spread of the 15 null runs. It is a")
    print("            rough yardstick, not a significance test: it ignores the")
    print("            AUC's own fold-to-fold error, so it flatters the lift.")
    print(f"\nWrote {OUT / 'barrier_width_sensitivity.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
