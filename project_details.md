# What this project is

A working spec. The README says how to run things; this says what we are
building and why each piece exists. Keep it current — when something here
changes, add a dated entry to `decision-log.md` rather than quietly editing
this file.

---

## The idea

We train a model to predict short-term price moves in five ETFs. Then we score
that same model, on the same data, three different ways, and report how far
apart the three answers are.

The model is the vehicle. The spread between those three numbers is the result.

## The experiment

| Scheme | What it does | Role |
| --- | --- | --- |
| Standard 5-fold K-fold, contiguous | Ordinary cross-validation, no correction | **Control.** Wrong for this data, which is why we run it |
| Purged 5-fold + embargo | Removes training rows that overlap the test fold | **Primary.** Our real number |
| Walk-forward, expanding | Only ever trains on the past | **Corroboration.** Agreement means the result isn't an artifact of one split |
| Holdout 2023–2025 | Opened once, at the very end | **Final.** No tuning against it, ever |

Every one of these is labelled as such wherever it appears — in the code, on
the site, in the write-up. The control must never read as an honest result.

Contiguous, not shuffled — `sklearn.KFold`'s own default, and the harder test
for our claim, since shuffling the same folds would leak 88% of training rows
instead of 0.33% and hand us a flattering gap. One control, one headline; the
shuffled figure appears once as context. See decision-log.md, 2026-11-10.

## The data

Five ETFs: SPY, QQQ, IWM, TLT, GLD. Daily bars, 2005 to 2025, about 5,000 rows
each.

GLD launched in November 2004, which is why the range starts in 2005. That is
the binding constraint, not a round number.

One row is one trading day for one ETF: open, high, low, close, adjusted
close, volume. We need the **high and low** because the label depends on
whether a price level was touched during the day, not on where it closed.

Cached in `data/raw/` with a date stamp, committed, never re-downloaded.

## The label

Binary. For each day *t*:

1. Compute `sigma_t`, the standard deviation of daily log returns over the
   previous 20 days, using data up to and including *t*.
2. Set the barriers as a log move off the close of *t*:
   `upper = close_t × exp(+2.5 × sigma_t)` and
   `lower = close_t × exp(−2.5 × sigma_t)`. Log space, so the two are exact
   mirrors — see the 2026-11-01 entries in the log for why 2.5 and why logs.
3. Walk forward from *t*+1 to *t*+10 trading bars, checking each day's high and
   low against both barriers.
4. **Label = 1** if the upper barrier is touched first, **or if both are
   touched on the same bar** — daily bars cannot order them, and the upper
   barrier is the side we would have acted on. **Label = 0** if the lower
   barrier is touched first, or if neither is touched within ten bars.
5. Record **`t1`**, the bar the label resolved on — the barrier-touch bar, or
   *t*+10 bars if it timed out. Always a real trading date, never a calendar
   offset. Rows whose window runs off the end of the data are dropped rather
   than recorded as timeouts.

That `t1` column is not optional bookkeeping. Purging is defined in terms of
it, so get it right before anything else is built on top.

Barriers scale with volatility so a label means the same thing in a calm month
as in a crisis. A fixed ±2% would be unreachable in one regime and noise in
another, and the model would waste capacity learning around that.

The width also has to match the horizon. At ±1 sigma the median label resolved
in a single bar, which left nothing for purging to purge; ±2.5 sigma gives a
mean window of about 6 bars and a timeout rate around 27%.

## Features

All computed from information available on or before day *t*, then shifted one
bar before the model sees them.

| Feature | Definition |
| --- | --- |
| Trailing returns | Log return over 1, 5, 10, 20 days |
| Realized volatility | Std of daily log returns, 20-day window |
| Volatility of volatility | Std of the 20-day vol series, 20-day window |
| Distance from MA | `(close − MA) / MA` for 20, 50, 200-day averages |
| Volume z-score | Volume against its own 20-day mean and std |

Feature sets enter the trial counter when they are scored against each other,
not when a column is written. See the counter's note in `decision-log.md`.

## Models

All of them predict the same binary target, so the comparison is like for like.

1. **Buy and hold** — reference, not really a model.
2. **Moving-average crossover, one-bar lag** — the baseline. Simple, widely
   used, and honestly lagged so it only acts on information it would have had.
3. **Logistic regression** on the standardized features. The floor: if nothing
   beats it, nothing complex is worth keeping.
4. **Gradient-boosted trees** (`HistGradientBoostingClassifier`). Low
   signal-to-noise and a small effective sample favour trees over a network.
5. **Meta-labelling** — the crossover rule picks the direction, the model
   predicts only whether that direction's target is reached first.

No neural network. The constraint is usable sample size, not model capacity.

## The evaluation machinery

We write all of this ourselves. Not importing it is the point.

**Purging.** For a test fold spanning `[test_start, test_end]`, drop every
training row *i* whose label window `[t_i, t1_i]` overlaps that span. In
practice: drop if `t1_i >= test_start and t_i <= test_end`.

**Embargo.** Additionally drop training rows with `t_i` in
`(test_end, test_end + embargo]`. Prices just after a test block are still
correlated with it even when their label windows do not overlap. Default is
about 1% of the sample; we sweep 0 to 20 days and report the sensitivity.

**Average uniqueness.** For each calendar day, count how many label windows
span it — call that concurrency. For each label, uniqueness is the mean of
`1 / concurrency` across its own window. Average across labels, multiply by
the row count, and that is the effective sample size we report beside the raw
one.

Measured, pre-holdout: average uniqueness ≈ **0.15**, so ~4,300 rows per ticker
carry the information of roughly **660** independent observations. Mean
concurrency ≈ 7. Every standard error we report is read against that number,
not the row count.

**Walk-forward.** Expanding window. Train on everything up to a point, test on
the next block, slide forward.

**Holdout.** 2023-01-01 to 2025-12-31, fixed in `config.yaml` before any
modelling. Opened once.

**Tuning discipline.** All hyperparameter search happens inside the purged
scheme. Scaling, imputation and feature selection are fitted on each fold's
training rows and then applied to its test rows — never fitted on everything
first.

## Metrics

Labels run about 40/60, so accuracy flatters a model that never acts.

- **Precision** — primary. Acting wrongly costs money; skipping costs little.
- **F1** and **AUC** — reported alongside, always next to the majority-class
  rate so a number can be read in context.
- **Sharpe with its standard error** — Sharpe is an average over a standard
  deviation, so it has error bars, and on this much data they are wide.
- **Maximum drawdown, turnover** — what the headline hides.
- **Deflated Sharpe Ratio** — adjusts Sharpe for how many configurations we
  tried. Only as honest as the trial counter.

## The validity check

Generate synthetic series containing a signal we planted ourselves. Run the
purged scheme on it. Confirm it recovers the signal.

Without this, a weak result on real data is ambiguous — it could mean there
was no signal, or it could mean our code is broken. This is what makes a null
result a finding rather than a failure.

## What we report

1. The purged result, stated plainly, including if it is unremarkable.
2. The K-fold-to-purged gap, computed for every configuration, reported as a
   range rather than a single number.
3. The model comparison under the purged scheme.
4. Representative errors — cases where the model said act and was wrong.
5. The synthetic check.
6. The holdout number, opened once.

## The website

A control panel over the backtest result. It changes how the model is
evaluated, not what the strategy does.

Controls: purging on/off, embargo 0–20 days, cost 0–20 bps, execution lag
same-bar or next-open. Each change redraws the equity curve and the summary
statistics.

**Precompute carefully.** All four controls crossed is 1,764 combinations, and
shipping a full equity curve for each is hundreds of megabytes. Instead ship
the ~42 position series (purging × embargo) and apply cost and lag
arithmetically in the browser. Decide this before building.

## Build order

Each step depends on the one above it.

1. Triple-barrier labelling, with the `t1` column. **Unit-test it on a
   hand-made series where you know the answers.**
2. Average uniqueness and effective sample size.
3. `PurgedKFold` with embargo. Test that no training row overlaps its test fold.
4. Walk-forward splitter.
5. Feature pipeline, fitted inside folds only.
6. Models, scored under all schemes.
7. Vectorised backtester with cost as a parameter.
8. Sharpe, its standard error, drawdown, turnover.
9. Deflated Sharpe Ratio.
10. Synthetic-signal check. *(Could be done right after step 3 — worth it.)*
11. Precomputed JSON grid.
12. Site.
13. Open the holdout. Once.
14. Write-up.

Steps 1 and 3 are where the project lives. If they are wrong, nothing after
them means anything, so test them harder than feels necessary.

## Division of work

Both of us need to be able to derive purging and the Deflated Sharpe Ratio
from scratch. The viva asks each of us separately, and "that part was Aksh's"
is not an answer worth five marks.

Do not split it as one person codes and one person builds the site.

## Open questions

Live in `decision-log.md` at the bottom. The first one you will hit is whether
to adjust the high and low by the same factor as Adj Close before computing
barriers — that bites at step 1.