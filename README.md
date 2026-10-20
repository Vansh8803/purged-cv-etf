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

Nothing to run yet beyond the downloader. Build order is step-by-step in
[`project_details.md`](project_details.md#build-order); steps 1 and 3
(triple-barrier labelling with the `t1` column, and `PurgedKFold`) come before
anything else and get tested harder than feels necessary.

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
