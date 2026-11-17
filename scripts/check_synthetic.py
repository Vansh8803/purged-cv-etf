"""Run the whole machinery on data whose answer we already know.

    python scripts/check_synthetic.py

Without this, a flat result on real ETF data is ambiguous: no edge, or broken
code. This separates those.

Three things get measured, and all three matter.

**Does the pipeline invent direction?** On a pure random walk (`phi = 0`) the
return features must score chance. If they beat it, every later result is
worthless.

**Does the pipeline find a planted signal?** With AR(1) predictability in
returns, it must beat chance. The planted signal sits in `ret_1d` -- literally
yesterday's return -- so the model has an honest and known route to it.

**What is the floor when there is no signal at all?** Not 0.50. The label's
own construction leaks: `sigma_t` sets the barrier width, wide barriers are
less likely to be touched inside the horizon, an untouched barrier is a timeout
and a timeout is label 0. So `realized_vol_20d` predicts the label on a random
walk with no price predictability whatsoever. The floor is reported here
because a real-data AUC must be read against it rather than against 0.50.

**Every number is averaged over seeds, and that is not optional.** One run of
this at `phi = 0` can land anywhere between 0.49 and 0.62 on the full feature
set. The run-to-run spread is wider than the effect being measured, so a
single-seed check is noise with a verdict attached.
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

from src.cv import NaiveKFold, PurgedKFold, WalkForward
from src.evaluate import cross_val_scores, summarise
from src.features import design_matrix, make_pipeline
from src.purging import embargo_bars
from src.synthetic import synthetic_dataset

#: The planted signal's only honest route into the model. Isolated from the
#: volatility features so the direction channel can be read on its own.
RETURN_FEATURES = ["ret_1d", "ret_5d", "ret_10d", "ret_20d"]

N_SEEDS = 10
PHI_SIGNAL = 0.6          # comfortably above the detection threshold; see the sweep
BASE_SEED = 1


def score(phi, seed, cfg, scheme="purged", columns=None):
    lab, split = cfg["label"], cfg["split"]
    df, bar_index = synthetic_dataset(
        n_bars=3000, phi=phi, seed=seed,
        barrier_mult=lab["barrier_mult"], horizon=lab["horizon_days"],
        vol_lookback=lab["vol_lookback_days"],
    )
    X, y, t1 = design_matrix(df)
    if columns:
        X = X[columns]

    embargo = embargo_bars(len(t1), split["embargo_frac"])
    cv = {
        "control": NaiveKFold(t1, bar_index, split["n_folds"]),
        "purged": PurgedKFold(t1, bar_index, split["n_folds"], embargo),
        "walk-forward": WalkForward(t1, bar_index, split["n_folds"]),
    }[scheme]

    pipe = make_pipeline(LogisticRegression(max_iter=1000, random_state=cfg["seed"]))
    return summarise(cross_val_scores(pipe, X, y, cv))["auc"]


def over_seeds(phi, cfg, scheme="purged", columns=None):
    a = np.array([score(phi, BASE_SEED + s, cfg, scheme, columns)
                  for s in range(N_SEEDS)])
    return a.mean(), a.std(ddof=1), a.min(), a.max()


def line(label, stats):
    m, sd, lo, hi = stats
    print(f"  {label:<42}{m:>7.3f}{sd:>8.3f}{lo:>8.3f}{hi:>8.3f}")


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    print("Synthetic-signal check")
    print(f"barrier_mult={cfg['label']['barrier_mult']}  horizon={cfg['label']['horizon_days']}  "
          f"folds={cfg['split']['n_folds']}  {N_SEEDS} seeds\n")
    print(f"  {'':<42}{'mean':>7}{'sd':>8}{'min':>8}{'max':>8}")

    print("\n  NO SIGNAL (phi = 0), purged scheme")
    noise_ret = over_seeds(0.0, cfg, columns=RETURN_FEATURES)
    noise_all = over_seeds(0.0, cfg)
    line("return features only -> must be ~0.50", noise_ret)
    line("all 10 features -> the no-signal floor", noise_all)

    print(f"\n  PLANTED SIGNAL (phi = {PHI_SIGNAL}), purged scheme")
    sig_ret = over_seeds(PHI_SIGNAL, cfg, columns=RETURN_FEATURES)
    sig_all = over_seeds(PHI_SIGNAL, cfg)
    line("return features only -> must beat 0.50", sig_ret)
    line("all 10 features", sig_all)

    print(f"\n  PLANTED SIGNAL (phi = {PHI_SIGNAL}) under every scheme, all features")
    for scheme in ("control", "purged", "walk-forward"):
        line(scheme, over_seeds(PHI_SIGNAL, cfg, scheme=scheme))

    print("\n  POWER CURVE: smallest planted signal the purged scheme can see")
    print(f"  {'phi':>5}{'10-day carry':>14}{'returns-only AUC':>19}")
    for phi in (0.0, 0.2, 0.4, 0.6, 0.8):
        m, sd, _, _ = over_seeds(phi, cfg, columns=RETURN_FEATURES)
        carry = sum(phi ** k for k in range(1, cfg["label"]["horizon_days"] + 1))
        print(f"  {phi:>5.1f}{carry:>14.2f}{m:>13.3f} ± {sd:.3f}")

    print("\n  VERDICT")
    checks = [
        ("no invented direction signal", abs(noise_ret[0] - 0.50) < 0.03),
        ("planted signal recovered", sig_ret[0] - noise_ret[0] > 0.05),
        ("label leaks through volatility", noise_all[0] - noise_ret[0] > 0.02),
    ]
    for name, passed in checks:
        print(f"    [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"\n  Read real-data AUC against {noise_all[0]:.3f}, not 0.500.")

    return 0 if all(p for _, p in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
