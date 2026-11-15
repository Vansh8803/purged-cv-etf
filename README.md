# Purged Cross-Validation for ETF Backtests

ES 335 · pair project · Vansh Dhameliya (25110337), Aksh Hareshkumar Patel (25110017)

We train one model to predict short-term price moves in five ETFs, then score
that same model, on the same data, under three different cross-validation
schemes, and report how far apart the three answers are.

The model is the vehicle. The spread between those numbers is the result.

- **What we are building and why** — [`project_details.md`](project_details.md)
- **Every decision, dated** — [`decision-log.md`](decision-log.md)
- **Every tunable parameter** — [`config.yaml`](config.yaml)

This file only says how to run it.

---

## Setup

Python 3.13. One virtual environment per machine, never committed.

**PowerShell (Windows):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.lock.txt
```

**bash (WSL, Git Bash, Linux, macOS):**

```bash
python3 -m venv .venv
source .venv/bin/activate        # Git Bash on Windows: source .venv/Scripts/activate
python -m pip install -r requirements.lock.txt
```

Install from `requirements.lock.txt`, not `requirements.txt`. The two are
different on purpose:

| File | What it is |
| --- | --- |
| `requirements.txt` | The floors we actually require, with reasons implied by the spec |
| `requirements.lock.txt` | Exact pinned versions, from `pip freeze` |

Reproducibility is the whole subject of this project, so the numbers in the
write-up have to come from a stated set of versions. If you add a package,
install it, then regenerate the lock with `python -m pip freeze > requirements.lock.txt`
and commit both files in the same commit.

**Do not build the venv from inside WSL and then run it from Windows, or the
reverse.** A venv hard-codes its own absolute path into `Scripts/*.exe` and
`activate`, so the two are not interchangeable and the failure is confusing.
Pick one and stay there. `.venv/` is gitignored precisely so each of us can
pick differently.

Check it worked:

```powershell
python -c "import pandas, numpy, sklearn, yaml, yfinance; print('ok')"
```

## Data

Already downloaded and committed. Five ETFs — SPY, QQQ, IWM, TLT, GLD — daily
bars, 2005-01-03 to 2025-12-30, 5,282 rows each, no missing values, all on the
same NYSE calendar.

```
data/raw/SPY_2005-01-01_2025-12-31_dl2026-10-16.csv
```

The filename carries the download date because Yahoo restates adjusted prices
retroactively. **Do not re-run the download.** Every later script reads
`data/raw/` and never touches the network. If you genuinely need a refresh:

```powershell
python scripts/download_data.py --force
```

which writes a *new* date-stamped file rather than overwriting the old one, so
last week's numbers stay reproducible. Without `--force` the script skips
tickers that are already cached, which makes a plain run a no-op.

Each row is one trading day for one ETF: `Date, Adj Close, Close, High, Low,
Open, Volume`. The high and low are load-bearing — the label depends on whether
a price level was *touched* intraday, not on where the day closed.

## Running things

Features, then labels. Both read `data/raw/` and write `data/processed/`,
which is gitignored because it is regenerable — so run these two on a fresh
clone before anything else.

```powershell
python src/data_process.py        # OHLCV -> features, shifted one bar
python scripts/build_labels.py    # features -> barriers, label, t1
python scripts/report_uniqueness.py   # how much independent data we actually have
```

`build_labels.py` takes every parameter from `config.yaml` and prints a
per-ticker summary over pre-holdout rows only.

Then the validity check, which has to pass before any real-data number means
anything:

```powershell
python scripts/check_synthetic.py
```

It runs the real pipeline on synthetic series with a known answer, over ten
seeds. Exits non-zero if any of its three checks fail. It also reports the
**no-signal AUC floor, currently 0.544** — real-data AUC is read against that,
not against 0.500, because the label leaks through volatility. See
decision-log.md, 2026-11-10.

Then the model comparison:

```powershell
python scripts/run_models.py      # five models x three schemes x five tickers
```

It writes `outputs/step06_scores.csv` and `outputs/step06_folds.csv`, and
prints the AUC and precision tables the write-up is built on. Pre-holdout rows
only, purged against the holdout rather than merely filtered by date. AUC is
printed beside its lift over the 0.544 floor, because that is the number that
means something.

Then the strategy-level numbers:

```powershell
python scripts/run_backtest.py    # positions, costs, Sharpe, drawdown, DSR
```

It writes `outputs/step789_backtest.csv` and `outputs/step789_equity.csv`.
Read every Sharpe against its own standard error, which is about 0.24 on this
much data — wide enough that only buy-and-hold is distinguishable from zero.

Steps 1 to 10 are done. Then the two reporting items that the other scripts cannot stand in for:

```powershell
python scripts/run_embargo_sweep.py   # the gap as a RANGE, 0-20 bars
python scripts/find_errors.py         # representative errors, with dates
```

`run_embargo_sweep.py` closes reporting item 2 — the spec asks for the
control-to-purged gap "reported as a range rather than a single number", and
until the sweep existed there was only one embargo setting. It also writes
`outputs/positions_<ticker>.csv`, which is the position series step 11 needs:
shipping positions rather than equity curves lets the browser apply cost and
execution lag arithmetically, per the 2026-10-14 decision.

`find_errors.py` closes reporting item 4. A gap number contains no errors in
it, so this is the only output that can satisfy it.

Remaining build order is step-by-step in
[`project_details.md`](project_details.md#build-order): the precomputed grid
(11), the site (12), opening the holdout (13) and the write-up (14).

Tests, once there are any:

```powershell
python -m pytest -q
```

## The holdout

`2023-01-01` to `2025-12-31`, declared in `config.yaml` on 2026-10-14, before
any modelling. It stays closed until step 13.

Opened once. No tuning against it, ever, including "just looking".

## Layout

```
config.yaml            every parameter that could be fiddled with
project_details.md     the working spec — what we are building and why
decision-log.md        dated decisions, reversals, and the trial counter
requirements.txt       dependency floors
requirements.lock.txt  exact pins — install from this one
data/raw/              cached OHLCV, committed, never re-downloaded
scripts/               one-off utilities; download_data.py lives here
```

## A note on the trial counter

The Deflated Sharpe Ratio we report is only as honest as the count of
configurations we tried, which lives at the top of `decision-log.md`.
Increment it whenever you evaluate a new combination — feature set, barrier
width, horizon, embargo length, model, hyperparameter grid point. It is easy
to forget and it invalidates the headline number when you do.
