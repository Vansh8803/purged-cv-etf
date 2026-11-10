# Decision log

Every choice, surprise and reversal, dated, written as it happens rather than
reconstructed in November. It feeds three things: the limitations section of
the write-up, the viva, and the honest trial count that the Deflated Sharpe
Ratio depends on.

**Rules.** One entry per decision, and record the *reason*, not just the
outcome. If a decision is reversed, add a new entry — never edit an old one.
Attribute honestly: who decided what, so the viva can be answered by whoever
is asked.

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

### 2026-11-03 · A test block's reach is its maximum t1

Decided (Aksh): when PurgedKFold purges around a test fold, that fold's span is
`[first test row's t, the largest t1 among the test rows]`, not
`[first test row's t, last test row's t]`.

A test label resolves up to `horizon` bars after its own row, so the stretch of
history a fold occupies runs past its last row. A training row starting in that
tail -- after the final test row, but on or before the last test label's
resolution date -- has a label window overlapping a test label. Measured by row
dates alone it survives the purge while holding part of the answer, and the
embargo would hide that rather than fix it: at embargo 0 the leak comes back in
full, which is precisely the setting the sensitivity sweep runs.

`src/purging.py` exposes this as `block_span()` rather than leaving it inline,
because the walk-forward splitter at step 4 faces the identical choice and the
two must not diverge.

### 2026-11-03 · Embargo bounds, and what sets its length

Decided (Aksh): the embargo drops training rows starting in
`(test_end, test_end + embargo]` -- exclusive at the start, inclusive at the
end, counted in trading bars rather than calendar days, and clipped at the end
of the sample rather than running off the index.

Exclusive at the start because `test_end` is already purged by the overlap rule
above, so including it again would be double counting. Inclusive at the end so
that an embargo of 1 actually withholds one bar; the off-by-one the other way
would make a 1-bar embargo a silent no-op, and the sweep starts at the low end
where that would matter most. Counted in bars for the same reason `t1` is: a
calendar offset lands on weekends and holidays, which are not rows.

The reason usually given for an embargo is serial correlation, which is true but
vague. The concrete one here is that our features include 20-day rolling
windows, so a training row just after a test fold has features computed from
bars inside it. That also gives the length a principled floor: at least the
longest feature lookback. `embargo_frac` of 0.01 gives 43 bars on the
pre-holdout sample, comfortably above the 20 that the rolling features need.
The 200-day moving average cannot be covered by any embargo we could afford and
belongs in limitations.

### 2026-11-03 · Correcting the shuffled-control leakage figures

Measured again (Vansh, reviewing step 3). The open question below quoted 99.9%
of training rows leaking under a shuffled control, against 0.5% contiguous.
Neither reproduces. On SPY, pre-holdout, 4,302 rows, horizon 10, 5 folds:

| control | train rows overlapping their test fold | test rows touched |
| --- | --- | --- |
| contiguous | 0.33% | 1.30% |
| shuffled | 88.5% | 99.98% |

The 99.9% came from measuring scattered folds with the interval test, the
`[min t, max t1]` form `block_span` uses. That form is right for a contiguous
block and meaningless for a shuffled one: once folds are scattered the interval
spans nearly the whole sample, so almost every training row counts as
overlapping it whether or not it shares a window with any test label. Counted
pairwise -- each training row against each test window, which is the actual
definition -- it is 88.5%, and stable at 88 to 90% across four seeds.

Why 88% rather than ~99%: a row's window averages about 6 bars, so it overlaps
roughly a dozen neighbours rather than the full 21 a 10-bar window either side
would give, and `0.8^12` leaves about 7% of rows untouched by any test fold.

None of this changes the conclusion, and the open question stands as the most
urgent thing outstanding. 0.33% against 88.5% is a 250-fold difference, and it
still decides what the headline number means. Recorded because the log is what
the write-up quotes, and a number that cannot be reproduced is worse there than
in a pull request.

Purging removes 0.33% of training rows from a contiguous control, matching the
contiguous leak exactly -- as it must, since purging removes precisely the
overlapping rows and nothing else.

### 2026-11-10 · Step 4: walk-forward splitter, and why its embargo does nothing

Added (Vansh): `WalkForward` in `src/cv.py`. The sample is cut into
`n_splits + 1` contiguous blocks; fold *i* trains on blocks 0..*i* and tests on
block *i+1*. One more block than folds, because the first is only ever trained
on. It goes through the same `purge_mask` as everything else.

Two things worth recording.

**Purging is still needed even though we only train on the past.** The obvious
reading of walk-forward is that training on history makes it leak-proof. It does
not: the last training rows before a test block have label windows running up to
10 bars forward, straight into the block. On the toy case the purge removes
exactly those 10 rows, which is now asserted. Training on the past prevents
training on *later rows*; it does nothing about a label that resolves later.

**The embargo is a no-op here, and that is a property rather than an
oversight.** The embargo drops training rows starting in
`(test_end, test_end + embargo]`. An expanding window has no training row
starting after its test block, so nothing is ever dropped. We kept the
parameter so the two splitters are interchangeable in the sweep, and the test
suite pins the no-op.

The consequence lands in the write-up, not the code: **walk-forward is flat
across the 0-to-20 embargo sweep by construction.** A flat line next to
PurgedKFold's sloping one looks like a bug or a null finding, and is neither.
If we wanted walk-forward to respond to the sweep it would need a gap withheld
*before* each test block, which is a different mechanism from de Prado's
embargo and would need its own justification. We are not adding one.

### 2026-11-10 · Differential test for the labelling code

Added (Vansh): `brute_force` in `tests/test_labels.py`, a plain-loop
reimplementation of the triple-barrier label, plus 30 cases checking the
vectorised `compute_labels` against it -- 25 random series and 5 deliberately
non-square (rows x horizon) shapes.

The vectorised version is fast but has two traps a reader cannot see: `argmax`
returns 0 on an all-False row, which is indistinguishable from a touch on the
first bar, and column *k* means bar *t+k+1*. The brute-force version has
neither, and is correct by inspection. Diffing them turns "is the matrix logic
right" into "do the two agree", which a test can actually answer.

The random series use barriers at +/-0.5% against ~3% daily ranges, far tighter
than our real 2.5 sigma. That is on purpose: at the real width a same-bar tie is
rare enough that random testing would almost never exercise the tie rule, which
is the branch most likely to be broken by a future edit.

The non-square shapes guard a specific bug: with `rows == horizon` the
comparison matrix is square, so a transposed broadcast would still run and
would compare row *t*'s barrier against bar *k*'s price.

### 2026-11-10 · AI-assistance disclosure removed from the log's rules

Changed (Vansh): the rules block at the top of this file claimed "the course
requires disclosing what AI assistance was used for, and this file is that
record". It does not. The professor's position is that we need to be able to
explain our own project, which is what the viva tests and what the crosswise
write-up in the work split is designed to guarantee.

The sentence was written into this file early and then followed in the
2026-10-14 and 2026-10-21 entries, which is how an unverified claim about the
course became a standing rule. Those two entries are left alone, per the rule
about never editing an old entry. The rules block is not an entry, so the
sentence is removed there and replaced with plain attribution -- who decided
what -- which is what the viva actually needs.

### 2026-11-10 · The control is contiguous K-fold. One control, one headline

Decided (Vansh), closing the open question: the control is **standard 5-fold
K-fold with contiguous folds**, and it is the only control the results tables
carry. `NaiveKFold` in `src/cv.py`, defaulting to `shuffle=False`.

Why contiguous and not shuffled:

- `project_details.md` says "standard 5-fold K-fold". `sklearn.KFold` does not
  shuffle by default, so that phrase already means contiguous. Picking the
  shuffled variant would mean quietly redefining our own spec.
- It is the harder test for our own claim. Shuffled K-fold leaks 88% of
  training rows and would hand us a huge, flattering gap. A control chosen
  because it makes our method look good is a strawman, and it is the first
  thing worth attacking in a viva.

What this costs us, stated plainly: **the headline gap will be small.** At our
label geometry purging removes 0.33% of training rows from a contiguous
control, so the K-fold-to-purged difference in score will be modest and may be
inside noise. That is the honest answer and it is the one we report.

It also sharpens what the project actually found, which is better than the gap
we originally expected:

> Fold geometry dominates leakage at this label length. Our labels resolve in
> about 6 bars, so with contiguous folds only the rows at four fold boundaries
> overlap a test fold at all -- 0.33%. Shuffle the same folds and the identical
> scheme leaks 88%. Purging drives both to exactly 0.00%, and it is cheap
> insurance whose value depends almost entirely on how the folds were cut.

The shuffled figure is kept as **one sentence of context**, not a parallel
results track. `NaiveKFold(shuffle=True, seed=...)` exists so that sentence is
reproducible, and the seed comes from `config.yaml`. It does not get its own
row in the model comparison, its own equity curve, or its own gap number.

`leakage()` moved into `src/cv.py` with tests. These figures were quoted wrong
once already (2026-11-03) and the write-up quotes them, so they are computed by
tested code rather than by a script that no longer exists. Measured on SPY,
4,302 pre-holdout rows, embargo 43 bars:

| scheme | train rows leaked | test rows touched |
| --- | --- | --- |
| control, K-fold contiguous | 0.33% | 1.30% |
| control, K-fold shuffled | 88.05% | 99.91% |
| purged 5-fold + embargo | 0.00% | 0.00% |
| walk-forward, expanding | 0.00% | 0.00% |

### 2026-11-10 · Scope: one number per question, not a matrix of options

Decided (Vansh): where a choice has been made and recorded, the write-up reports
the chosen branch and mentions the alternative in a sentence. It does not report
both branches as parallel results.

The reason is that the opposite had started to happen by accident. Several
decisions had arrived at "run both and report the pair" -- two controls, a
21-point embargo sweep, cost and lag crossed on the site -- and each was cheap
on its own. Together they multiply: five tickers times three schemes times two
controls times a sweep is a table nobody reads and a result nobody can state in
one sentence. A project with one clear claim beats one with six hedged ones, and
the rubric rewards sound reasoning rather than coverage.

What this does not touch: steps 1 and 3, the synthetic check, and the holdout
discipline. Those four are the project, per the work split's never-descope list,
and the decision log keeps every alternative we considered regardless of what
the write-up carries. The log is the audit trail; the write-up is the argument.

### 2026-11-10 · Step 5: preprocessing fitted in-fold, and what the model may see

Added (Vansh): `src/features.py`. Two things, and the distinction is the step.

The *features* are still computed globally by `data_process.py`, and that is
not a leak. Every one is a deterministic rolling function of its own past,
shifted one bar, so a row's values do not change with which fold it lands in.
Nothing is estimated from the sample.

The *preprocessing* is the part that had to move. A scaler learns column means
and standard deviations; fitted on the whole matrix, every fold's test rows
contribute to the numbers used to transform them. `make_pipeline` returns an
sklearn Pipeline rather than a pre-transformed matrix, so the only way to use
it is to call `fit` inside a fold. The canary test spikes a test-fold row to
1e9 and asserts the fitted scale does not budge.

`FEATURES` is an explicit list of ten, and `EXCLUDED` spells out what is
withheld and why, because "everything except label and t1" would quietly admit
all of it. Two exclusions are worth naming:

- `log_return` is the return *at* t, unshifted. `ret_1d` is the same number
  shifted, and is the one that belongs in the matrix.
- `sigma` is sigma_t, unshifted, and it *sets the barrier width*. Handing it
  over would break the one-bar rule and tell the model how far price must
  travel for the label to fire. `realized_vol_20d` is sigma shifted, and is the
  honest version. This turned out to matter more than expected -- see below.

No feature selection. Ten features against an effective sample near 700 does
not need it, and a selector is another thing to fit in-fold and another
configuration to count.

### 2026-11-10 · Step 10: the synthetic check passes, and found three things

Added (Vansh): `src/synthetic.py` and `scripts/check_synthetic.py`. Synthetic
series go through the *real* pipeline -- `engineer_features`,
`compute_barriers`, `compute_labels`, the real splitters, the real in-fold
preprocessing. One knob: `phi`, the AR(1) coefficient on daily returns. At
`phi = 0` there is nothing to find; above it the predictability is real and
sits in `ret_1d`, so the model has a known honest route to it.

All three checks pass. Each was a surprise.

**1. The no-signal floor is not 0.50. It is about 0.544.**

The label's own construction leaks. `sigma_t` sets the barrier width; wide
barriers are less likely to be touched inside ten bars; an untouched barrier is
a timeout; and a timeout is label 0. So trailing volatility predicts the label
on a pure random walk with no price predictability whatsoever. Measured on
phi = 0, ten seeds:

| features | AUC |
| --- | --- |
| return features only | 0.503 ± 0.028 |
| all ten features | 0.544 ± 0.040 |

`corr(realized_vol_20d, timeout) = +0.20`. Of the rows that resolved by
touching a barrier, 50.9% went up -- direction is a coin flip, exactly as a
random walk demands. The whole 0.044 comes from predicting *timeout*, not
direction.

This is a consequence of `timeout_is_negative: true` merging two different
outcomes into label 0: "went down" and "went nowhere". The second is
predictable from volatility and the first is not.

**We are not redesigning the label.** The binary target was decided on
2026-10-14 for reasons that still hold, and a three-class target would break
comparability across the five models. What changes is how the number is read:
**real-data AUC is reported against 0.544, not against 0.500**, and the write-up
says why. The synthetic check existing is what makes that possible; without it
we would have reported an AUC of 0.55 on real data as a modest edge when it is
the floor.

**2. One synthetic run proves nothing. The run-to-run spread is wider than the
effect.**

At phi = 0 on the full feature set, individual seeds landed between 0.489 and
0.623. The first version of this check used a single seed, read 0.571, and I
very nearly wrote up a structural floor of 0.57 from it. Everything is now
averaged over ten seeds and reported with its spread. The same point applies to
every real-data number we report: five folds on an effective sample near 700
has error bars wide enough to swallow most effects we might claim.

**3. The check's power is bounded, and that bounds what a null result means.**

Returns-only AUC against planted signal strength, ten seeds each:

| phi | 10-day carry | AUC |
| --- | --- | --- |
| 0.0 | 0.00 | 0.503 ± 0.028 |
| 0.2 | 0.25 | 0.513 ± 0.028 |
| 0.4 | 0.67 | 0.540 ± 0.020 |
| 0.6 | 1.49 | 0.610 ± 0.014 |
| 0.8 | 3.57 | 0.733 ± 0.014 |

Detection starts around phi = 0.4. Below that the planted signal is inside the
noise. So a flat result on real ETFs means "no AR(1)-style edge of phi >~ 0.4",
not "no edge". That is a far more defensible sentence than "we found nothing",
and it is the sentence the synthetic check buys us.

The three schemes score within 0.001 of each other at phi = 0.6 (control 0.605,
purged 0.605, walk-forward 0.589), which is the contiguous-control result from
2026-11-10 showing up again: with contiguous folds there is almost nothing to
purge, so purging changes almost nothing. On synthetic data where we know the
truth, that is reassurance rather than a disappointment.

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
