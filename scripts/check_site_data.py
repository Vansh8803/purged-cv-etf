"""Does the site compute the same numbers as the Python modules?

    python scripts/check_site_data.py

The site re-implements the backtest in JavaScript so that cost and execution
lag can be applied in the browser instead of being precomputed. That is the
2026-10-14 decision, and it buys a 475 KB file instead of hundreds of
megabytes -- but it means the same arithmetic now exists twice, in two
languages, and nothing stops the two drifting apart.

This is the guard. It ports the JavaScript in `site/index.html` line for line
into Python, runs it against `site/data.json`, and compares the result with
`src/backtest.py` and `src/metrics.py` computed from scratch. If they
disagree, the site is lying to whoever reads it, and that is a worse failure
than a wrong number in a notebook because it looks authoritative.

Also checks what the data contract promises: that the run-length encoding
round-trips, that every series the interface can ask for exists, and that no
bar after the declared holdout boundary is in the file.

Exits non-zero on any mismatch, so it can gate a commit.
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

from src.backtest import backtest, positions
from src.cv import NaiveKFold, PurgedKFold, WalkForward
from src.evaluate import oof_predictions
from src.features import design_matrix, load_labelled, pre_holdout
from src.metrics import max_drawdown, sharpe, sharpe_se
from src.models import model_zoo

PERIODS = 252
TOL = 1e-4  # generous: the file stores returns to 6 dp on purpose


def decode_rle(runs, n):
    """The site's `decodeRLE`, in Python. Runs alternate starting from zero."""
    out = np.zeros(n)
    i, v = 0, 0
    for run in runs:
        out[i:min(i + run, n)] = v
        i += run
        v = 0 if v else 1
    return out


def browser_stats(pos_in, returns, cost_bps, lag):
    """The site's `run()` and `stats()`, in Python."""
    n = len(returns)
    pos = np.zeros(n)
    if lag:
        pos[lag:] = pos_in[:-lag]
    else:
        pos[:] = pos_in

    traded = np.abs(np.diff(np.concatenate([[0.0], pos])))
    net = pos * returns - traded * (cost_bps / 1e4)
    equity = np.cumprod(1.0 + net)

    m, sd = net.mean(), net.std(ddof=1)
    sr = 0.0 if sd == 0 else m / sd
    return {
        "sharpe": sr * np.sqrt(PERIODS),
        "sharpe_se": 0.0 if sd == 0 else np.sqrt((1 + 0.5 * sr ** 2) / n) * np.sqrt(PERIODS),
        "max_drawdown": float((1 - equity / np.maximum.accumulate(equity)).max()),
        "total_return": float(equity[-1] - 1.0),
        "turnover": float(traded.mean() * PERIODS),
        "exposure": float(pos.mean()),
    }


def python_stats(pos, log_return, cost_bps):
    bt = backtest(pos, log_return, cost_bps=cost_bps)
    return {
        "sharpe": sharpe(bt["net_return"]),
        "sharpe_se": sharpe_se(bt["net_return"]),
        "max_drawdown": max_drawdown(bt["net_return"]),
        "total_return": float((1.0 + bt["net_return"]).prod() - 1.0),
        "turnover": float(bt["traded"].mean() * PERIODS),
        "exposure": float(bt["position"].mean()),
    }


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    path = ROOT / "site" / "data.json"
    if not path.exists():
        print(f"!! {path} missing — run scripts/build_site_data.py first")
        return 1

    d = json.loads(path.read_text(encoding="utf-8"))
    holdout_start = pd.Timestamp(cfg["split"]["holdout_start"])
    folds, seed = cfg["split"]["n_folds"], cfg["seed"]
    failures = []

    print(f"site/data.json — {path.stat().st_size / 1024:,.0f} KB, "
          f"{len(d['dates'])} bars, built {d['meta']['built']}")

    # --- the holdout boundary
    last = pd.Timestamp(d["dates"][-1])
    print(f"\nholdout: last bar {last.date()}, holdout opens {holdout_start.date()}")
    if last >= holdout_start:
        failures.append(f"data.json contains {last.date()}, inside the holdout")
    else:
        print("  clean — nothing from the holdout is in the file")

    # --- every series the interface can ask for
    want = ({"control_kfold", "walk_forward"}
            | {f"purged|{e}" for e in d["meta"]["embargos"]})
    missing = [f"{tk}/{m}/{k}"
               for tk in d["meta"]["universe"] for m in d["meta"]["models"]
               for k in want - set(d["tickers"][tk]["positions"][m])]
    print(f"\ncoverage: {len(want)} series x {len(d['meta']['models'])} models "
          f"x {len(d['meta']['universe'])} tickers")
    if missing:
        failures.append(f"{len(missing)} series missing, e.g. {missing[:3]}")
    else:
        print("  complete")

    # --- the arithmetic, on one configuration per scheme
    print("\narithmetic: browser vs src/backtest.py + src/metrics.py")
    for ticker in d["meta"]["universe"]:
        df = load_labelled(ticker)
        bars = df.index[df.index < holdout_start]
        X, y, t1 = design_matrix(pre_holdout(df, holdout_start, df.index))
        log_return = df["log_return"].reindex(bars)
        rets = np.array(d["tickers"][ticker]["returns"])

        schemes = {
            "control_kfold": (NaiveKFold(t1, bars, folds), "control_kfold"),
            "purged|0": (PurgedKFold(t1, bars, folds, 0), "purged|0"),
            "purged|20": (PurgedKFold(t1, bars, folds, 20), "purged|20"),
            "walk_forward": (WalkForward(t1, bars, folds), "walk_forward"),
        }
        for model in ("logistic", "gbt"):
            for name, (cv, key) in schemes.items():
                oof = oof_predictions(model_zoo(seed)[model], X, y, cv)
                pos_py = positions(oof["pred"] == 1, t1, bars)
                pos_site = decode_rle(d["tickers"][ticker]["positions"][model][key],
                                      len(rets))

                if not np.array_equal(pos_site, pos_py.to_numpy()):
                    failures.append(f"{ticker}/{model}/{name}: positions differ")
                    continue

                for cost in (0, 10):
                    a = browser_stats(pos_site, rets, cost, 0)
                    b = python_stats(pos_py, log_return, cost)
                    worst = max(abs(a[k] - b[k]) for k in b)
                    if worst > TOL:
                        bad = max(b, key=lambda k: abs(a[k] - b[k]))
                        failures.append(
                            f"{ticker}/{model}/{name} @{cost}bps: {bad} "
                            f"browser {a[bad]:.6f} vs python {b[bad]:.6f}")
        print(f"  {ticker}: checked")

    print()
    if failures:
        print(f"FAILED — {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("PASSED — the site and the Python modules agree, the run-length")
    print("encoding round-trips exactly, every series exists, and no holdout")
    print("bar is present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
