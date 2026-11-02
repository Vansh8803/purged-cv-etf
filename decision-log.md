# Decision log

Every choice, surprise and reversal, dated, written as it happens rather than
reconstructed in November. It feeds three things: the limitations section of
the write-up, the viva, and the honest trial count that the Deflated Sharpe
Ratio depends on.

**Rules.** One entry per decision, and record the *reason*, not just the
outcome. If a decision is reversed, add a new entry — never edit an old one.
Attribute honestly: the course requires disclosing what AI assistance was used
for, and this file is that record.

---

## Current settings

Update this block whenever an entry below changes it.

| Setting | Value | Set on |
| --- | --- | --- |
| Track | ES 335, pair | 2026-10-14 |
| Team | Vansh Dhameliya (25110337), Aksh Hareshkumar Patel (25110017) | 2026-10-14 |
| Title | Purged Cross-Validation for ETF Backtests | 2026-10-14 |
| Universe | SPY, QQQ, IWM, TLT, GLD | 2026-10-14 |
| Date range | 2005-01-01 to 2025-12-31 | 2026-10-14 |
| Bar frequency | Daily | 2026-10-14 |
| Target | Binary: profit target reached before stop loss | 2026-10-14 |
| Horizon | 10 trading days | 2026-10-14 |
| Barrier width | ±2.5 × 20-day realized volatility, in log space | 2026-11-01 |
| Holdout | 2023-01-01 to 2025-12-31, UNOPENED | 2026-10-14 |
| Primary CV | Purged 5-fold, embargo ≈1% of sample | 2026-10-14 |
| Control CV | Standard 5-fold K-fold | 2026-10-14 |
| Third scheme | Walk-forward, expanding window | 2026-10-14 |
| Repo | `purged-cv-etf` | 2026-10-21 |

## Trial counter

The DSR needs the number of configurations actually tested. Increment this
every time a combination is **scored** — feature set, barrier width, horizon,
embargo length, model type, hyperparameter grid point. Undercounting makes the
headline number worthless.

What counts is a configuration we got a performance number out of and could
therefore have selected on. Building a feature, or comparing barrier widths on
label statistics like uniqueness or timeout rate, produces nothing to select
on and is not a trial. The counter starts moving at step 6, when models are
first scored.

**Configurations evaluated so far: 0**

---

## Log

### 2026-10-14 · Subject of the project

Decided (Vansh): the project is about evaluation methodology, not about finding
a profitable strategy. We build a model, then measure how much the reported
score changes between a careless test and a leakage-safe one.

Reason: the rubric gives 20 of its 80 shared points to evaluation design, more
than any other criterion, and explicitly says a modest or negative result earns
full credit when the reasoning is sound. A pure prediction project is graded on
an accuracy that is not achievable on this data.

### 2026-10-14 · The naive scheme is a control, not a mistake

Decided (Vansh, after questioning whether running a known-wrong method is
defensible): standard K-fold is run deliberately, labelled as a control
everywhere it appears.

Reason: we cannot claim purging matters without measuring what it changes.
Risk accepted: a grader skimming the code could mistake it for incompetence,
so every appearance of it must carry the word "control".

### 2026-10-14 · Findings will not rest on the gap alone

Changed (Claude flagged, Vansh accepted): the proposal originally promised only
the K-fold-versus-purged gap as the result. Now it promises three things — the
honest holdout number stated plainly, the gap measured across every
configuration, and a synthetic-data check.

Reason: the gap alone leaves 15 rubric points for baseline and method selection
untouched, and 10 points for "representative errors" unreachable, because a gap
number has no errors in it.

### 2026-10-14 · Universe is five ETFs

Decided: SPY, QQQ, IWM, TLT, GLD. Not individual stocks.

Reason: survivorship bias. A list of stocks silently excludes everything that
went bankrupt. An ETF's price history already contains those companies from
when they were in the index. GLD launched in November 2004, which is the
binding constraint and the reason the range starts in 2005 — not an arbitrary
round number.

### 2026-10-14 · Ken French Data Library dropped

Reversed (Vansh questioned it, Claude confirmed it was wrong): a second data
source was in the draft and has been removed.

Reason: two independent problems. It publishes portfolio *returns*, so it has
no high or low column and our barrier label cannot be computed on it at all.
And the survivorship problem it was supposed to solve is already solved by
using ETFs. It was in the draft to make the data section look rigorous, which
is not a reason.

### 2026-10-14 · Target is binary, not three-class

Changed (Claude flagged inconsistency, Vansh accepted): the label is
"profit target reached before stop loss" — yes or no. A timeout counts as no.

Reason: the draft had logistic regression and the trees predicting three
classes while meta-labelling predicted two, and then promised to report all of
them side by side. Precision is not comparable across different targets. One
target for all models fixes it.

### 2026-10-14 · Holdout declared

Decided: 2023-01-01 to 2025-12-31. Not opened until everything else is done.
This file and config.yaml are the dated evidence that it was fixed first.

Reason: a holdout chosen after seeing results is just another thing you tried.

### 2026-10-14 · Purged CV written by us, not imported

Decided: no mlfinlab. We read it as reference and write our own PurgedKFold.

Reason: it is roughly fifty lines, 5 marks are for explaining our own choices
in the viva, and importing the one thing the project is about would be absurd.

### 2026-10-14 · No neural network

Decided: logistic regression, then HistGradientBoostingClassifier, then
meta-labelling on top.

Reason: the binding constraint is usable sample size after label overlap, not
model capacity. The course does not require deep learning.

### 2026-10-14 · Synthetic sanity check added

Added (Claude suggested, Vansh accepted): run the purged scheme on generated
data containing a signal we planted, and confirm it recovers it.

Reason: without it we cannot distinguish "purging removed a false edge" from
"our pipeline never worked". It converts a null result from a failure into a
finding.

### 2026-10-14 · Cross-sectional overlap acknowledged

Added (Claude flagged): the five ETFs share calendar dates, so rows overlap
across tickers as well as across time. Average uniqueness as usually computed
only handles the time dimension.

Still undecided: whether we pool all five into one model or fit per ticker.

### 2026-10-14 · Proposal submitted

Submitted to the portal. Fields: research question, problem and claim, data
plan, baseline and methods, evaluation plan, likely challenge, track-specific
plan.

AI assistance used so far: Claude for drafting and editing proposal prose,
for challenging the framing, and for catching the three-class/binary
inconsistency and the Ken French problem. All decisions above were made by us.

### 2026-10-21 · Repository created, named `purged-cv-etf`

Decided (Vansh): the work became a git repository today, named for the method
rather than the vehicle. The folder was renamed from `etf-backtest` to match.

Reason: the project is about evaluation methodology, not about backtesting
ETFs, and the repository name is the first thing a reader sees. `etf-backtest`
described the means and implied a strategy project. Renaming the directory
invalidated the virtual environment, which bakes its own absolute path into
`Scripts/*.exe`, so it was rebuilt — see the next entry.

Kept out of the repository deliberately: the venv, the Chan PDF (third-party
copyright, and not ours to redistribute), and the planning notes, which are
scheduling rather than a project artefact.

### 2026-10-21 · Dependencies pinned, not just floored

Added (Claude proposed, Vansh accepted): `requirements.lock.txt`, exact
versions from `pip freeze`, alongside the existing `requirements.txt` floors.
The README instructs installing from the lock.

Reason: every number we report is conditional on the library versions that
produced it, and reproducibility is the subject of this project rather than a
side concern. The floors in `requirements.txt` are also much looser than what
is actually installed — `pandas>=2.0` resolves to 3.0.6 today, a major version
with changed defaults, and `yfinance>=0.2.40` resolves to 1.7.0. Two of us on
two machines installing from floors would not have been running the same code.

Rule going with it: a new package means regenerating the lock and committing
both files together.

### 2026-10-21 · GLD needs no price adjustment at all

Observed (Claude checked, during setup): in the cached data GLD's `Adj Close`
equals `Close` to floating-point exactness on every one of its 5,282 rows. The
other four differ, as dividend-paying funds should.

Reason it matters: this narrows the open OHLC-adjustment question rather than
answering it. GLD holds bullion and distributes nothing, so its adjustment
factor is 1.0 throughout and the question of whether to scale High and Low by
`Adj Close / Close` cannot change a single GLD barrier. Whatever we decide
bites on SPY, QQQ, IWM and TLT only. Useful as a test case: a bug in the
adjustment path will show up as a difference between GLD and the rest, which
is a cheap thing to assert.

Still undecided. Not a decision, an observation that makes the decision
smaller.

### 2026-10-21 · Work split drafted

Drafted (Claude proposed at Vansh's request, pending Aksh's agreement): a
division of the build order between the two of us, with the rule that neither
of us owns purging or the Deflated Sharpe Ratio alone.

It lives outside the repository, in `../planning/`, because it is scheduling
rather than a project artefact and will be stale within a fortnight. The open
question below stays open until Aksh has actually agreed to it.

AI assistance this session: Claude for auditing the environment, for the
lockfile and README, for the GLD observation above, and for drafting the work
split. The naming decision and the acceptance of each change were ours.


### 2026-10-24 · High and Low scaled by Adjusted Close factor

We will synthesize an adjusted high and adjusted low by multiplying the raw High and Low columns by the factor `(Adj Close / Close)` before computing barrier touches.

Mixing raw intraday prices with a volatility threshold `sigma_t` derived from adjusted returns is a trap. Without scaling, a scheduled dividend drop would trigger a false lower-barrier stop-loss on SPY, QQQ, IWM, and TLT. While proportional scaling is a mathematical approximation of intraday dividend effects, it allows us to cleanly vectorize the label generation. The alternative—running strictly on raw prices and writing a discrete cash-management module to handle dividend reinvestment—pulls engineering time away from the evaluation machinery. GLD distributes nothing, so its adjustment factor is exactly 1.0; it will serve as the unit test for our scaling logic.

### 2026-10-24 · Per-ticker models chosen over pooled modeling

We will train and evaluate five separate models rather than concatenating all five ETFs into a single 25,000-row master dataset.

Cross-sectional data leakage. Because the ETFs share calendar dates, pooling them means a test fold evaluating QQQ in March 2018 would leave SPY's concurrent March 2018 data in the training set, allowing the model to cheat by reading the macro market's simultaneous reaction. Preventing this would require our `PurgedKFold` to drop the overlapping dates across all tickers simultaneously, carving massive 10-day holes out of the entire universe and severely cannibalizing the effective sample size. We sacrifice the predictive power of a larger dataset to guarantee mathematically clean, strictly 1D time-based purging.

### 2026-11-01 · Barrier width raised from 1.0 to 2.5 sigma

We changed `barrier_mult` from 1.0 to 2.5 in `config.yaml`.

At 1.0 the label was broken in a way that would have quietly wrecked the whole
project. A barrier one daily sigma wide is narrower than a typical day's
high-to-low range, so almost everything touched immediately: the median
holding period was **1 bar**, 52% of labels resolved on t+1, and the vertical
barrier bound on under 1% of rows. Label windows that short barely overlap, so
purging would have had almost nothing to purge and the K-fold-to-purged gap --
the headline result -- would have come out near zero. That would have been an
artifact of the barrier width, not a finding about cross-validation.

It also contradicted our own metric choice. `project_details.md` says labels
are imbalanced and that is why precision is primary; at 1.0 the measured
balance was 48-52% on all five ETFs.

We compared 2.5 against sqrt(10) = 3.162, the width where the barrier matches
a typical 10-day move. Averaged over the five ETFs, pre-holdout:

| mult | label 1 | timeout | mean t1-t | avg uniqueness | effective N |
| --- | --- | --- | --- | --- | --- |
| 1.0 | 0.520 | 0.9% | 1.9 bars | 0.362 | 1558 |
| 2.5 | 0.405 | 27% | 5.6 bars | 0.160 | 690 |
| 3.162 | 0.304 | 44% | 6.7 bars | 0.135 | 579 |

2.5 wins on two counts. It keeps about 19% more effective sample, and sample
size is the binding constraint on this project rather than model capacity. And
at 3.162 nearly half of all labels are timeouts, meaning half the data carries
no information about direction at all.

The cost is real and goes in the limitations section: effective sample size
drops from 1558 to 690. Wider barriers mean longer label windows, more overlap,
and less independent data. That is the correct trade -- the overlap is the thing
we are studying -- but it is a cost.

### 2026-11-01 · A same-bar tie counts as label 1

When the high clears the upper barrier and the low breaks the lower barrier on
the same bar, we label it 1.

Daily bars cannot tell us which level was touched first, so this is a choice
rather than a measurement. We take 1 because the upper barrier is the side we
would actually have acted on -- the position is opened for the profit target,
and a bar that reached it counts as reaching it. The alternative, calling every
ambiguous bar 0, is more conservative but it systematically throws away exactly
the volatile bars where the signal lives.

This matters more now than it used to. At the old 1.0 width ties were rare; at
2.5 the barriers are wide enough that a bar touching both is a genuinely large
day, and those are not rows we want to mislabel by default. Worth revisiting
with intraday data, which we do not have.

### 2026-11-01 · t1 counts trading bars, and unresolved rows are dropped

Two fixes to the labelling code, both in `src/labels.py`.

**t1 was in calendar days.** The timeout date was `t + pd.Timedelta(days=10)`,
so for a Thursday t it returned a Sunday -- not a trading date at all. Purging
is *defined* as an overlap test on `[t, t1]`, so a t1 that is not a bar makes
every purge decision slightly wrong. It is now a positional offset of 10 bars
in the index, and there is a test asserting every t1 is a real bar. At the old
barrier width this only hit 0.9% of rows; at 2.5 it would have hit 28%.

**The last rows were labelled as timeouts.** Shifting the future off the end of
the series gives NaN, NaN comparisons are False, so the final rows registered
no touch and fell through to "timeout, label 0". They were not timeouts, they
just had no future to look at. We now drop any row whose t1 would land past the
end of the data. A row that genuinely touches a barrier inside the data we have
is kept, so this drops 2 to 10 rows per ticker rather than a blanket 10.

### 2026-11-01 · Barriers computed in log-return space

`sigma_t` is the standard deviation of daily *log* returns, so the barriers are
a log move off day t's close:

    upper = close_t * exp(+2.5 * sigma_t)
    lower = close_t * exp(-2.5 * sigma_t)

`project_details.md` said "cumulative return from the close of t", which could
have been read as `close * (1 +/- 2.5 * sigma)`. In simple-return space the up
move and the down move are different sizes in log terms, so the label would
inherit a small upward bias that has nothing to do with the market. Log space
makes the two barriers exact mirrors: `upper * lower == close^2`, which is now
asserted in a test.

Everything else in the pipeline already used log returns -- the features, the
volatility, the vol-of-vol -- so this just removes the one place it was
ambiguous. Labels and barriers are written to
`data/processed/{ticker}_labelled.csv` by `scripts/build_labels.py`, which
reads every parameter from `config.yaml` rather than hard-coding it.

### 2026-11-01 · Step 2 done: we have about one-sixth of the data we thought

`src/uniqueness.py` computes concurrency, per-label uniqueness and effective
sample size. Numbers below are pre-holdout, per ticker, at barrier_mult 2.5.

| ticker | rows | avg uniqueness | effective N |
| --- | --- | --- | --- |
| SPY | 4302 | 0.1563 | 672 |
| QQQ | 4304 | 0.1585 | 682 |
| IWM | 4302 | 0.1548 | 666 |
| TLT | 4307 | 0.1477 | 636 |
| GLD | 4303 | 0.1523 | 656 |

So roughly 4,300 rows per ticker carry about as much independent information as
660 unrelated observations. Mean concurrency is about 7, meaning a typical
trading day sits inside seven different label windows at once.

This is the number that justifies the project. If labels were independent,
standard K-fold would be defensible and there would be nothing to measure. At
0.15 uniqueness, almost every training row overlaps the test fold it is being
scored against, which is exactly the leakage purging removes.

It also changes what we can claim. Every standard error in the write-up has to
be read against ~660, not ~4,300 -- error bars are about 2.5x wider than the
row count suggests. Model choice follows too: the 2026-10-14 decision against a
neural network was argued from usable sample size, and 660 effective
observations makes that argument much stronger than it looked at the time.

Two checks that it works. Max concurrency comes out at exactly 11 on all five
tickers, which is the theoretical ceiling -- a window [t, t+10] spans 11 bars,
so at most 11 windows can cover one bar. And the five tickers land within 0.011
of each other, which is what we would expect from the same horizon applied to
five different price series.

One API decision worth recording: `average_uniqueness` requires the trading
calendar as an explicit argument rather than inferring it from the labels. A
window's length is only meaningful in bars, and the (t, t1) dates alone do not
pin the calendar down -- t1 can point past the last labelled row, and gaps in
that date set silently shorten every window measured against it. A test caught
this, and the function now raises instead of guessing.

---
## Open questions

Move these into the log as dated entries once decided.

- **JSON size for the website.** Precomputing all 2 × 21 × 21 × 2 = 1,764
  slider combinations as full equity curves is hundreds of megabytes. Plan:
  ship the ~42 position series (purging × embargo) and apply cost and
  execution lag in the browser. Confirm before building.
- **Is the DSR a live number or a fixed one?** It depends on the total trial
  count, which is a property of the whole project, not of a slider position.
  Probably display it fixed, with the per-configuration Sharpe live.
- **Final feature list.** Currently about eight. They only enter the trial
  counter once feature sets are scored against each other at step 6.
- **Split of work between Vansh and Aksh.** Both must be able to derive
  purging and the DSR — the viva asks each of us separately. Do not split it
  as "one codes, one does the website".
