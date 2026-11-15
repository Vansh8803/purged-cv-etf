# Purged Cross-Validation for ETF Backtests

ES 335 · pair project · Vansh Dhameliya (25110337), Aksh Hareshkumar Patel (25110017)

We train one model to predict short-term price moves in five ETFs, then score
that same model, on the same data, under three different cross-validation
schemes, and report how far apart the three answers are.

The model is the vehicle. The spread between those numbers is the result.

- **What we are building and why** — [`project_details.md`](project_details.md)
- **Every decision, dated** — [`decision-log.md`](decision-log.md)
- **Every tunable parameter** — [`config.yaml`](config.yaml)

**The website is the submission.** It carries the full reasoning — question,
data, preprocessing, models, evaluation, results, error analysis and
limitations — with one interactive panel:

- **[vansh8803.github.io/purged-cv-etf](https://vansh8803.github.io/purged-cv-etf/)**

This repository is what lets you verify it. This file says how to run it.

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

```powershell
python scripts/check_floor_real.py
```

measures that same floor on real bars instead of synthetic ones, by
randomising the sign of each daily return — which keeps the volatility path
and destroys direction. It prints the comparison and changes nothing; whether
to adopt it is an open question.

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

Then the site (steps 11 and 12):

```powershell
python scripts/build_site_data.py    # the precomputed grid -> docs/data.json
python scripts/check_site_data.py    # does the browser agree with the modules?
cd docs; python -m http.server 8000  # then open http://localhost:8000
```

A local server is needed because browsers block `fetch` from `file://`. The
page says so if you open it directly.

`build_site_data.py` ships the position series rather than equity curves, one
per scheme and embargo, run-length encoded. Crossing all four controls would be
1,764 equity curves and hundreds of megabytes; this is 475 KB, and cost and
execution lag are applied arithmetically in the browser. That was decided on
2026-10-14 and the file is the evidence it was the right call.

`check_site_data.py` is the guard that matters. The site re-implements the
backtest in JavaScript, so the same arithmetic now exists twice in two
languages. That script ports the JavaScript back into Python, runs it against
`docs/data.json`, and compares with `src/backtest.py` and `src/metrics.py`
computed from scratch. It exits non-zero on any disagreement, and it also
checks that the run-length encoding round-trips and that no holdout bar is in
the file. Run it after anything upstream changes.

Remaining build order is step-by-step in
[`project_details.md`](project_details.md#build-order): the precomputed grid
(11), the site (12), opening the holdout (13) and the write-up (14).

Tests, once there are any:

```powershell
python -m pytest -q
```

## Data access

Everything needed to reproduce the results is already in this repository. No
credentials, no API key, no download.

- `data/raw/` holds the five cached CSVs, committed on purpose. Each is one ETF
  of daily OHLCV from Yahoo Finance via `yfinance`, 2005-01-03 to 2025-12-30,
  5,282 rows, named with the date it was downloaded: `2026-10-16`.
- `data/processed/` and `outputs/` are **gitignored**, because everything in
  them is regenerable from `data/raw/`. The commands under *Running things*
  rebuild them in order.
- `docs/data.json` **is** committed, because the website cannot be served from a
  clone without it.

**Do not re-download.** Yahoo restates adjusted prices retroactively, so a
fresh download would silently change numbers computed from the old one. That is
why the filenames carry a download date and why `scripts/download_data.py`
skips tickers that are already cached. If you genuinely need a refresh, pass
`--force` and it writes a *new* dated file rather than overwriting.

To verify the cached data is what we say it is:

```powershell
python -c "import pandas as pd; d=pd.read_csv('data/raw/SPY_2005-01-01_2025-12-31_dl2026-10-16.csv'); print(len(d), d.Date.min(), d.Date.max(), int(d.isna().sum().sum()))"
```

## Reproducing the results

Every number on the website comes from one of these, in this order. Timings are
from a laptop; the two model scripts dominate.

| command | produces | roughly |
| --- | --- | --- |
| `python src/data_process.py` | `data/processed/*_features.csv` | seconds |
| `python scripts/build_labels.py` | `data/processed/*_labelled.csv` | seconds |
| `python scripts/report_uniqueness.py` | effective sample size table | seconds |
| `python scripts/check_synthetic.py` | the no-signal floor and power curve | ~2 min |
| `python scripts/run_models.py` | `outputs/step06_*.csv` | ~3 min |
| `python scripts/run_backtest.py` | `outputs/step789_*.csv` | ~5 min |
| `python scripts/run_embargo_sweep.py` | `outputs/embargo_sweep.csv`, positions | ~12 min |
| `python scripts/find_errors.py` | `outputs/representative_errors.csv` | ~2 min |
| `python scripts/open_holdout.py` | `outputs/step13_holdout.csv` | ~1 min |
| `python scripts/build_site_data.py` | `docs/data.json` | seconds |
| `python scripts/build_site_findings.py` | the website's tables | seconds |

Then `python -m pytest -q` for the 161 tests, and
`python scripts/check_site_data.py`, which re-implements the website's
JavaScript in Python and asserts it agrees with `src/backtest.py` and
`src/metrics.py`.

All randomness is seeded from `seed` in `config.yaml`, so repeated runs give
identical numbers.

**Splits are code, not prose.** The exact splits are `PurgedKFold`,
`WalkForward` and `NaiveKFold` in [`src/cv.py`](src/cv.py), all built on the one
overlap rule in [`src/purging.py`](src/purging.py), and the holdout boundary is
`purged_holdout_mask` in the same file. The test suite asserts the properties
rather than the outputs: `tests/test_cv.py` checks that no surviving training
row's label window overlaps its test fold, and `tests/test_purging.py` pins the
embargo as exclusive at the start and inclusive at the end.

## AI-use disclosure

We used an AI coding assistant (Anthropic's Claude, via Claude Code) throughout
this project. It was used for:

- implementing and refactoring the code in `src/` and `scripts/` from our
  specifications;
- writing and extending the test suite;
- code review, including checking our formulas against the papers they come
  from and probing the test suite for gaps;
- drafting documentation — this README, the prose in `decision-log.md`, and the
  website copy;
- exploratory analysis, such as parameter sweeps and summarising results into
  tables.

All design decisions were ours, and each is recorded with its date and reasoning
in [`decision-log.md`](decision-log.md): the research question, the universe,
the binary target, the barrier width, the choice of control, per-ticker models,
and how the holdout would be reported. We reviewed the outputs, and we can
derive the two methods the project turns on — purging and the Deflated Sharpe
Ratio — independently of the code.

Citations for the methods themselves follow.

## Citations

**Methods.** The purging and embargo construction, the triple-barrier label,
average uniqueness and meta-labelling all follow Marcos López de Prado,
*Advances in Financial Machine Learning* (Wiley, 2018), chapters 3 to 7. We
read it as a reference and wrote our own implementations; `mlfinlab` and
similar libraries are deliberately not used, since the method is the subject of
the project.

The Deflated Sharpe Ratio, including the expected-maximum formula with its
Euler–Mascheroni weighting, is David H. Bailey and Marcos López de Prado, *The
Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and
Non-Normality*, Journal of Portfolio Management 40(5), 2014.

The standard error of the Sharpe ratio is Andrew W. Lo, *The Statistics of
Sharpe Ratios*, Financial Analysts Journal 58(4), 2002. We use the IID case and
say so in the limitations.

Ernest P. Chan, *Quantitative Trading*, informed the backtesting discipline —
costs as a parameter, turnover, and not trusting a Sharpe ratio without its
error bars. The book itself is not redistributed here.

**Data.** Yahoo Finance, accessed through `yfinance`.

**Software.** numpy, pandas, scikit-learn, scipy, PyYAML, yfinance, pytest.
Exact versions are pinned in `requirements.lock.txt`; install from that rather
than from `requirements.txt`.

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
docs/                  the website (GitHub Pages serves this folder)
outputs/               generated tables and position series; gitignored
```

## A note on the trial counter

The Deflated Sharpe Ratio we report is only as honest as the count of
configurations we tried, which lives at the top of `decision-log.md`.
Increment it whenever you evaluate a new combination — feature set, barrier
width, horizon, embargo length, model, hyperparameter grid point. It is easy
to forget and it invalidates the headline number when you do.
