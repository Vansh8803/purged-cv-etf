# Decision log

We write down every choice, surprise and reversal here on the day it happens,
because by November we would not remember why we did any of it. We use this
log for three things: the limitations we have to report, answering questions
about our own work, and keeping an honest count of how many configurations we
tried, which the Deflated Sharpe Ratio depends on.

**How we keep it.** One entry per decision, and we always write down the
*reason* rather than just what we chose. If we change our minds we add a new
entry instead of editing the old one, so the reversal stays visible. We say
who decided what, so either of us can answer for any part of it. Our AI-use
disclosure lives in the README, which is where the submission instructions ask
for it.

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
| Holdout | 2023-01-01 to 2025-12-31, **OPENED 2026-11-15** | 2026-11-15 |
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

**Configurations evaluated so far: 5** — see the 2026-11-14 entry.

---

## Log

### 2026-10-14 · Subject of the project

Vansh set the direction: we are studying evaluation methodology, not trying to
find a profitable strategy. We build a model, then measure how much its
reported score moves between a careless test and a leakage-safe one.

We went this way because the rubric gives 20 of its 80 shared points to
evaluation design, more than it gives anything else, and it says outright that
a modest or negative result earns full credit when the reasoning is sound. If
we chased prediction instead, we would be graded on an accuracy this data
cannot deliver.

### 2026-10-14 · The naive scheme is a control, not a mistake

Vansh asked whether running a method we already know is wrong can be
defended. We decided it can, as long as we run it deliberately and label it a
control wherever it appears.

We cannot claim purging matters without measuring what it changes, and that
needs an uncorrected baseline to compare against. The risk we are accepting is
that somebody skimming our code mistakes it for incompetence, so every place it
turns up has to carry the word "control".

### 2026-10-14 · Findings will not rest on the gap alone

Our proposal originally promised one thing only: the gap between the K-fold
and purged scores. A review flagged that as too thin and Vansh agreed, so we
now promise three things — the holdout number stated plainly, the gap measured
across every configuration, and a check on synthetic data.

The gap on its own would have left 15 rubric points for baseline and method
selection untouched, and made the 10 points for representative errors
impossible to earn, because a single gap number contains no errors to show.

### 2026-10-14 · Universe is five ETFs

We chose SPY, QQQ, IWM, TLT and GLD rather than individual stocks.

Our reason is survivorship bias. Any list of stocks we picked today would
silently exclude every company that went bankrupt along the way. An index
fund's price history already contains those companies from the years they were
in the index. We start in 2005 because GLD launched in November 2004, so that
date is a binding constraint rather than a round number we liked.

### 2026-10-14 · Ken French Data Library dropped

Our draft listed a second data source, Vansh questioned whether we needed it,
and once we looked properly we dropped it.

It failed for two separate reasons. It publishes portfolio *returns*, so there
is no high or low column and we cannot compute our barrier label on it at all.
And the survivorship problem it was meant to solve, we had already solved by
using index funds. It was really there to make our data section look rigorous,
which is not a reason to keep it.

### 2026-10-14 · Target is binary, not three-class

Our label is now a single question with a yes or no answer: was the profit
target reached before the stop loss? A timeout counts as no.

A review caught an inconsistency we had missed, and Vansh accepted the fix. Our
draft had logistic regression and the trees predicting three classes while
meta-labelling predicted two, and then promised to report all of them in one
table. Precision does not mean the same thing across different targets, so that
table would have been meaningless. Giving every model the same target fixes
it.

### 2026-10-14 · Holdout declared

We are holding back 2023-01-01 to 2025-12-31, and we will not open it until
everything else is finished. This file and `config.yaml` are our dated evidence
that we fixed those dates before seeing any results.

We are being this careful because a holdout chosen after seeing results is not
a holdout at all. It is just one more thing we tried.

### 2026-10-14 · Purged CV written by us, not imported

We are not importing mlfinlab. We will read it as a reference and write our
own `PurgedKFold`.

It is about fifty lines of code, five marks depend on us explaining our own
choices, and importing the one thing our project is about would be absurd.

### 2026-10-14 · No neural network

We will use logistic regression, then `HistGradientBoostingClassifier`, then
meta-labelling on top of those. No neural network.

What limits us here is how much usable data we have once label overlap is
accounted for, not how much capacity our model has. The course does not ask for
deep learning either.

### 2026-10-14 · Synthetic sanity check added

We are adding a check that came out of a review and that Vansh accepted: we
generate data containing a signal we planted ourselves, run the purged scheme
on it, and confirm it finds what we put there.

Without it, a flat result would be ambiguous. We could not tell "purging
removed a false edge" from "our pipeline never worked at all". With it, a null
result becomes a finding rather than a failure.

### 2026-10-14 · Cross-sectional overlap acknowledged

A review pointed out something we had not thought about. Our five funds share
calendar dates, so rows overlap across funds as well as across time, and
average uniqueness as it is normally computed only handles the time dimension.

We have not decided yet whether to pool all five into one model or fit one per
fund.

### 2026-10-14 · Proposal submitted

We submitted the proposal to the portal, covering the research question, the
problem and our claim, the data plan, the baseline and methods, the evaluation
plan, the likely challenge and the track-specific plan.

We used AI assistance to draft and edit the prose and to challenge our framing.
Every decision above is ours.

### 2026-10-21 · Repository created, named `purged-cv-etf`

Vansh turned the work into a git repository today and named it for the method
rather than the vehicle. We renamed the folder from `etf-backtest` to match.

We are studying evaluation methodology rather than backtesting ETFs, and the
repository name is the first thing anyone reads. `etf-backtest` described our
means and implied we were building a strategy. Renaming the directory broke the
virtual environment, because a venv bakes its own absolute path into
`Scripts/*.exe`, so we rebuilt it — see the next entry.

We are deliberately keeping three things out of the repository: the venv, the
Chan PDF, which is third-party copyright and not ours to redistribute, and our
planning notes, which are scheduling rather than part of the project.

### 2026-10-21 · Dependencies pinned, not just floored

This came out of a review and Vansh accepted it: we now keep
`requirements.lock.txt` with exact versions from `pip freeze`, alongside the
floors we already had in `requirements.txt`. The README tells you to install
from the lock.

Every number we report depends on the library versions that produced it, and
reproducibility is the subject of this project rather than a side concern. Our
floors are also far looser than what we actually have installed: `pandas>=2.0`
resolves to 3.0.6 today, a major version with changed defaults, and
`yfinance>=0.2.40` resolves to 1.7.0. The two of us installing from those
floors on two machines would not have been running the same code.

The rule that goes with it: if either of us adds a package, we regenerate the
lock and commit both files together.

### 2026-10-21 · GLD needs no price adjustment at all

We noticed during setup that GLD's `Adj Close` equals its `Close` to
floating-point exactness on every one of its 5,282 rows. The other four differ,
which is what dividend-paying funds should do.

This narrows our open question about adjusting high and low rather than
answering it. GLD holds bullion and distributes nothing, so its adjustment
factor is 1.0 all the way through, and whether we scale High and Low by
`Adj Close / Close` cannot change a single GLD barrier. Whatever we decide only
bites on SPY, QQQ, IWM and TLT.

It does give us a cheap test. If we get the adjustment path wrong, it will show
up as a difference between GLD and the other four, and that is easy to assert.
We have not decided the question yet, we have only made it smaller.

### 2026-10-21 · Work split drafted

At Vansh's request we drafted a division of the build order between the two of
us, with one rule: neither of us owns purging or the Deflated Sharpe Ratio
alone. Aksh has not agreed to it yet.

We keep it outside the repository, in `../planning/`, because it is scheduling
rather than part of the project and it will be stale within a fortnight. The
open question below stays open until Aksh has actually said yes.

We used AI assistance for environment setup and for drafting and reviewing
documentation. The naming decision and the acceptance of each change were
ours.


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

Aksh decided how we measure a fold's span. When PurgedKFold purges around a
test fold, that span is `[first test row's t, the largest t1 among the test
rows]`, not `[first test row's t, last test row's t]`.

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

Aksh settled the embargo's bounds. It drops training rows starting in
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

Vansh added `WalkForward` to `src/cv.py`. We cut the sample into
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

Vansh added `brute_force` to `tests/test_labels.py`. It is a plain-loop
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

Vansh changed this. The rules block at the top of this file claimed "the course
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

Vansh closed our most urgent open question. Our control is **standard 5-fold
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

Vansh decided how we will write this up. Where we have made a choice and
recorded it, the write-up reports the branch we chose and mentions the
alternative in a sentence. It does not report both as parallel results.

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

Vansh added `src/features.py`. Two things live in it, and telling them apart is
what this step is about.

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

Vansh added `src/synthetic.py` and `scripts/check_synthetic.py`. Our synthetic
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

### 2026-11-10 · A sabotage run found a hole in the step 3 tests

Vansh added `test_block_span_uses_max_t1_not_the_last_row_even_when_sorted` to
`tests/test_purging.py`.

Found by deliberately breaking each core line and checking what the suite
caught. Replacing `test_t1.max()` with `test_t1.iloc[-1]` in `block_span` broke
exactly one test -- the scattered-block one added on 2026-11-03 -- and only
because that test passes an unsorted block. Nothing was asserting the sorted
case.

It should have been. `t1` is not monotonic: on SPY 841 of 5,056 rows resolve
earlier than the row before them, because a row that touches a barrier
immediately resolves sooner than a row that timed out ten bars ago. In the
final contiguous fold `max(t1)` is a week later than the last row's `t1`. So
the bug was live on real data in a sorted contiguous block, which is the only
shape `PurgedKFold` ever produces, and the suite would not have caught it.

The toy series in `test_cv.py` hid it: every label there resolves exactly
`horizon` bars later, so `max` and `last` always coincide. A uniform fixture
cannot exercise a bug about non-uniformity, which is worth remembering when the
step 6 fixtures get written.

Nothing was wrong in the shipped code -- `block_span` already used `max` -- but
the property was unprotected, and that is the same thing one edit later.

### 2026-11-14 · Step 6: the gap is zero, and the simple model wins

Aksh added `src/models.py`, `scripts/run_models.py` and `tests/test_models.py`.
We scored five models under three schemes on five tickers, pre-holdout, fitting
the preprocessing inside every fold. The numbers below are averaged over the
five tickers.

AUC, read against the **0.544 no-signal floor** from step 10 rather than 0.50:

| model | control | purged | walk-forward | lift vs floor |
| --- | --- | --- | --- | --- |
| buy_and_hold | 0.5000 | 0.5000 | 0.5000 | -0.044 |
| ma_crossover | 0.5131 | 0.5131 | 0.5108 | -0.031 |
| logistic | 0.5758 | 0.5755 | 0.5710 | **+0.032** |
| gbt | 0.5554 | 0.5558 | 0.5439 | +0.012 |
| meta_crossover_logistic | 0.5476 | 0.5475 | 0.5398 | +0.004 |

**The headline gap is zero.** Control minus purged is +0.0003 for logistic,
-0.0004 for the trees, against a per-fold standard error of about 0.017. The
gap is roughly fifty times smaller than the noise on the number it is measured
from. This is not a disappointment and it is not a null we have to apologise
for: it is the direct consequence of the contiguous-control decision, which
predicted it in writing on 2026-11-10 before any model was scored. Only 0.33%
of training rows leak in a contiguous control, so there was never room for the
score to move. **Fold geometry, not purging, is what decides how much a
careless backtest flatters itself at this label length.**

**Logistic beats gradient boosting**, 0.576 against 0.556, and beats it on
precision too, 0.493 against 0.455. The 2026-10-14 argument against more
capacity was that the effective sample is ~700, not that trees were wrong, and
this is that argument landing. The complex model did not earn its place, which
is itself the method-selection result rather than a failure to find one.

**Both rules sit below the floor.** Buy-and-hold scores 0.500 by construction
and the moving-average crossover 0.513, against a floor of 0.544. A model that
knew nothing except trailing volatility would beat our baseline. Worth saying
plainly in the write-up: the baseline is not a strawman we beat, it is a rule
that does essentially nothing on this target.

**Precision is where meta-labelling shows up.** 0.516 against always-act's
0.405, the best of the five, while its AUC is mid-table. That is the shape
meta-labelling is supposed to have -- it does not rank better, it declines to
act where the rule would have been wrong -- and precision is our primary metric
because acting wrongly costs money.

**Walk-forward agrees**, 0.571 against the purged 0.576 for logistic. Slightly
lower everywhere, which is expected since it trains on less data, and no
disagreement that would suggest the purged result is an artifact of one split.

Caveats that belong in limitations. The logistic lift of +0.032 is about 1.9
standard errors, which is suggestive and not significant. Reading it against
0.50 instead of 0.544 would have turned a marginal lift into an apparent 0.076
edge, which is exactly the error step 10 was built to catch.

**No hyperparameter search.** Settings are fixed, conservative, and chosen for
the sample size rather than tuned on it. A grid would multiply the trial count
the Deflated Sharpe Ratio depends on, for a model class the evidence already
says is too big for the data.

### 2026-11-14 · Trial counter moves to 5

Aksh set the counter to **5**, the five models in `model_zoo` scored per
ticker. Vansh has not confirmed it yet.

Per the counter's own rule, a trial is a configuration we got a performance
number out of and could therefore have selected on. The three schemes are
measurements of the same model rather than alternatives we would choose
between, so they do not multiply it. Each ticker gets its own model and its own
Deflated Sharpe Ratio, so the five tickers do not multiply it either. No
hyperparameter grid was run, so nothing is hidden there.

**Configurations evaluated so far: 5**

### 2026-11-14 · Review of step 6: three scoring fixes, and a better null

Reviewed (Vansh, on Aksh's step 6). Every number in the step 6 entry
reproduces, 121 tests passed, and the purging machinery came through clean. The
central conclusion -- the gap is zero because the control is contiguous -- is
correct and was predicted in writing beforehand. Four things about how the
numbers were *presented* were wrong, and all of them fed straight into the
write-up.

**1. Precision was averaged across folds, and the averaging inflated it.**

Our models act on as little as 11% of bars, and the per-fold action count
swings by an order of magnitude inside a single ticker -- 157, 22, 170, 225, 50
for logistic on SPY. An unweighted mean of per-fold ratios gives the 22-action
fold the same say as the 225-action one, and a small denominator is a noisy one
that reliably reads high.

| model | pooled | unweighted fold mean | inflation |
| --- | --- | --- | --- |
| logistic | 0.4532 | 0.4930 | +0.040 |
| gbt | 0.4435 | 0.4546 | +0.011 |
| meta_crossover_logistic | 0.4891 | 0.5160 | **+0.027** |

Worst for the model whose entire claim is precision. Meta-labelling's margin
over always-act drops from +0.111 to +0.084 -- still the best of the five, and
the conclusion survives, but a quarter smaller than reported. `summarise` now
sums `n_true_pos` and `n_acts` and divides once.

**2. The printed precision benchmark was the majority rate, which is the wrong
number.**

The table showed `majority_rate` = 0.596. Precision is read against the
**positive rate, 0.404** -- the number always-act scores by construction. At
40% positives those two differ by 0.19, and against 0.596 every model looks
like a failure. The step 6 entry had it right in prose and the code had it
wrong, which is the sort of disagreement that becomes an error in the write-up.

**3. Coverage was not reported at all.**

Precision alone rewards abstention: a model acting once and getting it right
scores 1.0. The pair has to travel together.

| model | coverage | pooled precision | vs benchmark |
| --- | --- | --- | --- |
| buy_and_hold | 100.0% | 0.405 | +0.001 |
| ma_crossover | 63.0% | 0.420 | +0.016 |
| logistic | **11.0%** | 0.453 | +0.049 |
| gbt | 26.2% | 0.444 | +0.040 |
| meta_crossover_logistic | **13.7%** | 0.489 | +0.085 |

**4. The gap's standard error was the wrong statistic, and the right one is
ten times tighter.**

`NaiveKFold(shuffle=False)` and `PurgedKFold` split the same rows the same way,
so their test folds are *identical* -- verified on all 125 of them -- and only
the training sets differ. The gap is therefore a paired comparison, and its
noise is the spread of per-fold differences:

| model | gap | paired s.e. | t | s.e. previously quoted |
| --- | --- | --- | --- | --- |
| logistic | +0.00027 | 0.00087 | **0.31** | 0.0168 |
| gbt | -0.00038 | 0.00202 | -0.19 | 0.0160 |
| meta | +0.00009 | 0.00039 | 0.24 | 0.0148 |

The conclusion is unchanged and better supported. "Fifty times smaller than the
noise" divided the gap by the noise on a different comparison; the honest line
is **gap +0.0003, paired standard error 0.0009, t = 0.31**. `paired_gap` in
`src/evaluate.py`, with a test showing the paired error collapsing to zero on
folds that move together while the unpaired one stays wide.

### 2026-11-14 · A null measured on real bars: the floor is 0.529, not 0.544

Vansh added `sign_flipped_ohlc` to `src/synthetic.py`, plus
`scripts/check_floor_real.py`.

`NO_SIGNAL_FLOOR = 0.544` came from a constant-volatility Gaussian random walk.
Real ETFs cluster their volatility and have fat tails, and the floor is
*produced* by a volatility channel, so whether that number transfers is a fair
question. It was worth checking before the write-up rested on it.

The test: multiply each daily return by a random +/-1. That leaves every
`|r_t|` untouched, so the volatility path survives -- rolling sigma correlates
**0.998** with the real one on SPY -- while all directional predictability is
destroyed. Intraday reach magnitudes carry over too, so the barrier-touch rate
and hence the timeout rate are preserved. An ordinary randomisation test.

| ticker | floor | sd |
| --- | --- | --- |
| SPY | 0.540 | 0.015 |
| QQQ | 0.525 | 0.016 |
| IWM | 0.520 | 0.021 |
| TLT | 0.523 | 0.034 |
| GLD | 0.535 | 0.013 |
| **pooled** | **0.529** | 0.021 (s.e. 0.0039, n=30) |

**The synthetic floor is conservative, which is the safe direction to be
wrong.** The real floor is 0.015 lower, so 0.544 *understates* logistic's lift:
+0.047 against the real null rather than +0.032. Nothing reported so far was
flattered by the error.

Three reasons to switch anyway: it is measured on the real feature
distribution at the real sample size through the real folds; it is per ticker,
and the 0.520-to-0.540 spread is hidden by a single constant; and its standard
error is 0.0039 against the synthetic's 0.040 seed spread, so it stops being
the dominant uncertainty in the lift. Combined with the paired standard error,
logistic's lift goes from "1.9 s.e., suggestive" to roughly 2.7.

**Not wired in.** Swapping the floor moves a headline number, so it needs both
of us. The script prints the comparison; `NO_SIGNAL_FLOOR` is untouched.

One thing that fell out and belongs in the write-up. Real lag-1 return
autocorrelation, pre-holdout: SPY -0.107, QQQ -0.089, IWM -0.077, TLT -0.024,
GLD -0.010. Step 10's power curve puts detection at about phi = +0.4. The real
structure is an order of magnitude smaller than the smallest signal the
pipeline can resolve, and the wrong sign for momentum besides. So the null
result is exactly what the instrument predicts rather than a surprise -- which
is a far better sentence than "we found nothing", and it is the one the
synthetic check was built to earn.

### 2026-11-14 · Steps 7, 8 and 9: the index wins, and the best classifier is the worst strategy

Aksh added `src/backtest.py`, `src/metrics.py`, `scripts/run_backtest.py`,
`oof_predictions` in `src/evaluate.py`, and tests for all of it. We take
out-of-fold
predictions under the purged scheme, turned into positions, costed, and
measured. Averaged over the five tickers, pre-holdout.

| model | Sharpe | SE | t | max DD | turnover | exposure | Sharpe @10bps | DSR @0 | DSR @10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| buy_and_hold | **0.480** | 0.242 | 1.99 | 0.514 | 0.1 | 1.00 | 0.480 | 0.820 | 0.778 |
| ma_crossover | 0.363 | 0.242 | 1.50 | 0.360 | 4.8 | 0.68 | 0.329 | 0.655 | 0.560 |
| logistic | 0.057 | 0.242 | 0.24 | 0.284 | 10.2 | 0.22 | **-0.058** | 0.282 | 0.138 |
| gbt | 0.410 | 0.242 | 1.70 | 0.358 | 18.9 | 0.51 | 0.259 | 0.714 | 0.478 |
| meta_crossover_logistic | 0.328 | 0.242 | 1.36 | **0.201** | 9.5 | 0.23 | 0.206 | 0.623 | 0.435 |

**Buy-and-hold has the best Sharpe, and no model beats it.** That is the
result. It is a modest and partly negative finding, and the rubric says in
writing that such a result earns full credit when the reasoning is sound.

**The best classifier is the worst strategy.** Logistic had the highest AUC in
step 6 at 0.576 and has the lowest Sharpe here at 0.057, turning negative at 10
bps. Ranking ability and money are different questions: AUC asks whether the
model orders bars correctly, and a strategy asks whether the bars it chose to
hold went up by more than the trading cost. A project that stopped at the AUC
table would have reported logistic as the winner. This is the single most
useful thing steps 7 to 9 added.

**Turnover is what does the damage.** The gradient-boosted trees trade 18.9
times a year and lose a third of their Sharpe between 0 and 10 bps, 0.410 to
0.259. Buy-and-hold trades 0.1 times a year and loses nothing. This is the cost
sensitivity the proposal promised, and it is doing real work rather than
decorating a table.

**Almost nothing is distinguishable from zero.** The standard error is 0.242
for every model, because at roughly 4,300 daily bars the error is set by the
sample and not by the strategy -- Lo's formula depends on the Sharpe only
through a term that is negligible when the per-period ratio is this small. Only
buy-and-hold reaches t = 2, and that is the index having risen, not a discovery.

**No Deflated Sharpe clears a convincing bar.** The luckiest of five skill-less
tries would be expected to show an annualised Sharpe of 0.25, which is half of
what buy-and-hold achieved and more than any learned model achieved. The best
DSR is buy-and-hold's 0.82 and the best learned model is the trees at 0.71,
falling to 0.48 at 10 bps. Nothing is near 0.95.

Honest caveats, for limitations:

- **The exposure column makes this not quite apples to apples.** Buy-and-hold
  is invested 100% of the time; logistic is invested 22%. A Sharpe comparison
  across different exposures rewards being in a rising market, and 2005-2022 on
  these five ETFs was mostly a rising market. Meta-labelling's drawdown of 0.20
  against buy-and-hold's 0.51 is the other side of that: less time in the
  market is less time exposed to the crash.
- **Four of five folds train partly on the future.** This is a cross-validated
  backtest, not a simulation of trading, and the equity curve is labelled as
  such. Walk-forward is the scheme that does not do this.
- **Lo's standard error assumes IID returns.** Daily returns are serially
  correlated and volatility clusters, so 0.242 understates the true error. It
  is still better than quoting no error at all, which is the usual practice.
- **N = 5 is what we admit to.** The DSR is only as honest as that count.

Design decisions worth recording, both in `src/backtest.py`:

**The holding period comes from the label.** A signal at the close of *t* opens
a position on the next bar and holds until `t1`. The model was trained to
answer "does the upper barrier fall first", so the trade the backtest runs is
the trade the label scores. A fixed ten-day hold would measure something the
model was never asked about.

**Overlapping signals do not stack.** Position is capped at one unit. Labels
overlap by construction -- the subject of the whole project -- so on most bars
several predictions are live, and stacking them would make leverage a function
of signal density and turnover meaningless. The cost is that a strong run of
signals looks like a single one.

### 2026-11-14 · Review of steps 7-9: exposure, not turnover, is what sinks logistic

Reviewed (Vansh, on Aksh's steps 7-9). Every number reproduces, 158 tests
passed, and the hard parts are right. Specifically checked and correct: the DSR
matches Bailey and Lopez de Prado (2014) equation 5 including the
Euler-Mascheroni weighting; `sharpe_se` is Lo (2002) for the IID case with the
per-period ratio inside it and the annualisation outside, and the 0.242 it
reports is analytically exactly `sqrt(252/4300)`; `positions` opens on the bar
after the signal and the difference-array bounds are safe at both ends of the
calendar.

**A check worth recording, because somebody will ask.** The label is an
*intraday* barrier touch while the backtest exits at the *close* of `t1`, so a
correct prediction need not be a winning trade. Measured: a label=1 row is a
winning close-to-close trade **99.4%** of the time, mean +2.67% against -1.40%
for label=0, a spread of **+4.06% per trade**. The label and the money are
aligned, and a perfect classifier would have made a fortune. That rules out the
obvious alternative explanation for the headline, so the headline is about the
models rather than about the label.

**The one correction: "turnover is what does the damage" is right for the trees
and wrong for logistic.**

At 0 bps there are no costs at all, and logistic is already at 0.057 against
buy-and-hold's 0.480. Turnover cannot explain a gap that exists before any
turnover is charged. The step 7-9 entry does flag exposure as a caveat, but
then attributes the damage to turnover in a heading, and the caveat is the
mechanism.

Added `exposure_matched_sharpe` in `src/metrics.py` to make it a number rather
than an apology: shuffle the position series, which holds the *count* of
invested bars exactly and destroys *which* bars they were, and average over 200
draws. That is what a strategy with this exposure and no selection skill
scores. Buy-and-hold validates the control -- at ~100% exposure there is
nothing to shuffle and its edge comes out -0.0008.

At 0 bps, averaged over the five tickers:

| model | exposure | matched benchmark | actual | selection edge | t |
| --- | --- | --- | --- | --- | --- |
| buy_and_hold | 1.00 | 0.481 | 0.480 | -0.001 | -0.54 |
| ma_crossover | 0.68 | 0.391 | 0.363 | -0.028 | -0.43 |
| logistic | 0.22 | 0.221 | 0.057 | **-0.164** | **-2.36** |
| gbt | 0.51 | 0.340 | 0.410 | +0.071 | 1.37 |
| meta_crossover_logistic | 0.23 | 0.228 | 0.328 | +0.099 | 0.84 |

The decomposition for logistic reads **0.480 -> 0.221 -> 0.057 -> -0.058**:
most of the fall is exposure, because being out of the market 78% of the time
forfeits the equity premium whatever you pick; then negative selection; then
cost. Three mechanisms, and the write-up should name all three rather than
blaming turnover for the lot.

**No model shows significant positive selection.** The trees at +0.071 and
meta-labelling at +0.099 are the right sign but t = 1.37 and 0.84, which is
nothing. The only result that clears two standard errors is **logistic's
negative edge**: it picks worse-than-random days, and it does so while having
the highest AUC of the five. That is a sharper version of Aksh's finding rather
than a contradiction of it -- ranking ability and profitability are different
questions, and here they point in opposite directions.

It also squares with step 6. Meta-labelling had the best precision there, 0.489
against the 0.404 base rate, and it has the best selection edge here. Two
different instruments agreeing on the same weak signal is worth a sentence in
the write-up, as long as that sentence says the signal is not significant.

**One limitation to add about the DSR.** `sr_std` is the spread of Sharpes
across five *structurally different* strategies, two of which -- buy-and-hold
and the crossover -- are fixed references rather than search outcomes. The
expected-maximum formula assumes N draws from a distribution of zero-skill
configurations. Using the observed spread of heterogeneous strategies instead
is a defensible approximation and a conservative one, since the spread is wider
than a true zero-skill distribution would be, which raises the benchmark and
lowers every DSR. Worth one sentence rather than a rework.

### 2026-11-15 · Representative errors: the models fail in crises, by construction

Vansh added `scripts/find_errors.py`, which writes
`outputs/representative_errors.csv`. This closes reporting item 4, which
nothing else could stand in for -- a gap number contains no errors in it.

An error is a day the model said act and the lower barrier fell first,
out-of-fold under the purged scheme, so every one is a day the model had
genuinely not been trained on.

**How often each model is wrong when it acts**, against the only honest
baseline, which is acting every single day:

| model | days acted | wrong | error rate | if always acting | better by |
| --- | --- | --- | --- | --- | --- |
| logistic | 2,370 | 1,260 | 0.532 | 0.595 | +0.063 |
| gbt | 5,640 | 3,132 | 0.555 | 0.595 | +0.040 |
| meta_crossover_logistic | 2,947 | 1,511 | 0.513 | 0.595 | **+0.082** |

Meta-labelling is again the best of the three, which is the third instrument
to say so after precision at step 6 and the selection edge at step 7-9. It is
still not significant, and the write-up should say all three agree and that
none of them clears a bar.

**The costliest errors are all the same two weeks.** The five worst days for
each model are October and November 2008, January 2009, and March 2020 --
the financial crisis and the Covid crash, and mostly IWM, the small-cap fund
with the widest swings. The single worst is a logistic signal on IWM for
2020-03-09 that lost 14.1% in three bars. Confidence on these was ordinary,
between 0.52 and 0.70: the model was not unusually sure, it was ordinarily
sure and catastrophically wrong.

**Errors concentrate where the barriers are widest.** Error rate by quintile
of trailing volatility on the day:

| volatility | logistic | gbt | meta |
| --- | --- | --- | --- |
| lowest | 0.481 | 0.495 | 0.497 |
| low | 0.486 | 0.490 | 0.449 |
| mid | 0.567 | 0.548 | 0.554 |
| high | 0.526 | 0.589 | 0.511 |
| highest | **0.641** | **0.602** | **0.622** |

This is not a coincidence and it is the most useful thing in the analysis. It
is the volatility channel from step 10 showing its other face. High volatility
means wide barriers, which means the label is more likely to time out, which
counts as 0 -- so the conditions in which the model is most likely to be wrong
are exactly the conditions its edge is built on. The edge and the failures have
the same cause. A model leaning on volatility to predict timeouts will act
confidently in precisely the regimes where a real move can blow through a wide
barrier in either direction.

By year, the pattern is the same: 2008 is the worst year for every model
(logistic 0.770), followed by 2009, 2022 and 2016. Distance from the 200-day
average gives a U shape -- errors are highest at both extremes, in strong
downtrends and strong uptrends, and lowest in the middle.

**What this means for the write-up.** The honest sentence is that these models
are least reliable exactly when it matters most, that this follows from the
label's construction rather than from a bug, and that it would be caught by any
practitioner before deployment. It also makes the case for the barrier width
being the project's most consequential single choice, which is where this all
started on 2026-11-01.

### 2026-11-15 · The embargo sweep: the gap is measurable, negligible, and not even sign-stable

Vansh added `scripts/run_embargo_sweep.py`, which writes
`outputs/embargo_sweep.csv` and `outputs/positions_<ticker>.csv`. This closes
reporting item 2, which asks for the gap "computed for every configuration,
reported as a range rather than a single number". Until now it existed at one
embargo setting, which is a single number -- the thing the spec says not to
report.

Every embargo from 0 to 20 bars, five models, five tickers, five folds, paired
against the control fold for fold. The embargo is swept on the purged scheme
only: the control has no embargo by definition, and walk-forward's is a
structural no-op already proven in the test suite.

**The gap as a range.** Units are AUC x 1000 throughout, because in AUC itself
every number is zeroes:

| model | min | max | spread | pooled mean | s.e. | t |
| --- | --- | --- | --- | --- | --- | --- |
| buy_and_hold | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | - |
| ma_crossover | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | - |
| logistic | -0.70 | -0.13 | 0.57 | -0.39 | 0.16 | -2.4 |
| gbt | -2.55 | +0.55 | 3.10 | -1.27 | 0.32 | -4.0 |
| meta_crossover_logistic | -0.18 | -0.03 | 0.15 | -0.11 | 0.06 | -1.8 |

The two rules come out at exactly zero, as they must: a rule is not fitted, so
it cannot be affected by which rows were withheld from training. That is a
useful check that the sweep is wired up correctly rather than a finding.

**Three things to say about this, and the third is the interesting one.**

*It is measurable.* Pool enough folds and the gap clears two standard errors.
With 525 paired differences per model you can detect almost anything.

*It is negligible.* The largest gap anywhere is about 1.3 thousandths of an AUC
point, against a purged AUC of 0.539 and a single fold's standard error of
about 17 thousandths. The effect is roughly forty times smaller than the noise
on one fold of the measurement it is drawn from. Statistically present,
practically absent -- which is a distinction worth making explicitly in the
write-up, because "significant" and "meaningful" are being used as synonyms by
everyone who quotes a p-value at a large sample.

*It is not sign-stable.* The gap's direction changes from ticker to ticker:

| ticker | logistic | gbt | meta |
| --- | --- | --- | --- |
| SPY | -2.61 | -2.36 | -1.01 |
| QQQ | +0.97 | -1.45 | -0.25 |
| IWM | -0.88 | -0.09 | +0.09 |
| TLT | +0.55 | +0.15 | +0.67 |
| GLD | +0.01 | -2.58 | -0.07 |

Three of logistic's five tickers go one way and two the other. This also
explains an apparent conflict with the step 6 table, which reported the gap as
**+0.0003** for logistic at the config embargo of 43 bars while the sweep comes
out slightly negative over 0 to 20. Both are right. The two scoring paths agree
to six decimals -- checked -- and the difference is simply that a quantity this
small wanders across embargo values and across tickers. **The sign of the gap
is not determined by our data**, and no version of this result should be quoted
as though it were.

**Also worth recording: the pooled sign is negative, meaning purging very
slightly *improves* AUC.** That is the opposite of the naive story, in which
leakage inflates the control. A hypothesis rather than a finding: with
contiguous folds the rows purged are the ones whose label windows straddle a
fold boundary, and a row needs an unusually long window to reach into the test
fold, so the purge disproportionately removes long-window timeout rows. Taking
a biased subset out of training may help more than the 0.33% of leaked rows
hurt. We are not claiming this, and it is far too small to matter either way.

**The purged AUC barely moves across the sweep.** Logistic reads 0.5763 at
embargo 0 and 0.5764 at embargo 20, wandering by less than a thousandth in
between. Withholding twenty extra bars after every test fold costs nothing
measurable, which is the sensitivity analysis the proposal promised and the
answer is that the result is insensitive.

**The position series for step 11 came free.** One fitting pass yields both the
AUC and the position series, so `outputs/positions_<ticker>.csv` now holds
4,312 bars by 115 columns -- five models by (control, walk-forward, and purged
at each of 21 embargos), strictly 0 or 1, named like
`logistic|purged_kfold|12`. Shipping positions rather than equity curves is the
2026-10-14 decision: cost and execution lag are arithmetic the browser can do.
The date range stops at 2022-12-30, so the holdout is untouched. Note for step
11: the two rules have 21 identical columns each, since a rule does not change
with the embargo, and they can be collapsed.

**Trial counter: unchanged, at 5.** A sensitivity sweep over one parameter is a
measurement of one configuration's stability, not a search for the best
setting. We report the range and do not pick from it. If an embargo is ever
selected on these numbers the counter moves by 21, and that would need to be
recorded here on the day it happens.

### 2026-11-15 · Effective sample size corrected: 662, not 690

Corrected (Vansh). The 2026-11-01 barrier-width entry quotes effective sample
sizes of 1558, 690 and 579 for barrier multiples of 1.0, 2.5 and sqrt(10).
Those came from throwaway code written before `src/uniqueness.py` existed, and
they also used a plain date filter on the holdout rather than
`purged_holdout_mask`. Recomputed with the tested module:

| barrier mult | label 1 | avg uniqueness | effective N | previously logged |
| --- | --- | --- | --- | --- |
| 1.0 | 0.534 | 0.3654 | 1575 | 1558 |
| 2.5 | 0.405 | 0.1539 | **662** | 690 |
| 3.162 | 0.304 | 0.1256 | 540 | 579 |

The decision does not change -- 2.5 still keeps about 23% more effective sample
than sqrt(10), which was the argument -- but the write-up quotes these numbers
and the log's own rule from 2026-11-03 is that a figure which cannot be
reproduced is worse in a write-up than in a pull request. `src/models.py` and
`src/features.py` said "roughly 700" and now say "roughly 660".

Per the never-edit rule the old entry stands; this is the correction.

### 2026-11-15 · Steps 11 and 12: the site, and the guard that keeps it honest

Vansh added `scripts/build_site_data.py`, `site/index.html` and
`scripts/check_site_data.py`, taking a step our work split assigns to Aksh --
see the caveat at the end of this entry.

**What is precomputed.** Only the position series, one per scheme and embargo,
plus each ticker's daily returns. The browser multiplies them together and
applies cost and execution lag on top. Crossing all four controls is
2 x 21 x 21 x 2 = 1,764 combinations, and a full equity curve for each is
hundreds of megabytes; that was decided against on 2026-10-14 and this is the
decision carried out.

**Run-length encoding, which made it small enough to stop worrying about.** A
position series is 4,312 bars but it is blocky, because a position is held for
about six bars. The median series has 117 runs instead of 4,312 bars, a 37-fold
reduction on the position data. The whole grid is **475 KB** -- 575 series
across five tickers, five models and 23 scheme/embargo combinations.

Two size decisions worth recording. The five ETFs share one NYSE calendar and
the labelled frames come out the same length, so the date axis is stored once
rather than five times; that alone was 220 KB of the first version, spent on
four exact copies. And returns are rounded to six decimal places, which is far
inside what any chart or statistic needs.

**The site labels the control as a control.** Selecting standard K-fold
replaces the status banner with a warning that it is wrong for this data and
run deliberately. That is the 2026-10-14 requirement that the control must
never read as an honest result, and a web page is the easiest place to break
it. The embargo slider disables itself for the control and for walk-forward,
because neither has an embargo -- the control by definition, walk-forward
because an expanding window has no training row after its test block.

**No charting library.** The equity curve is drawn on a canvas in about sixty
lines. A CDN dependency that fails during a viva is a worse outcome than
hand-rolled axes, and the page has no build step, no framework and no network
dependency at all beyond its own data file.

**"One bar late" is not a next-open fill, and the page says so.** The spec says
"same-bar or next-open". An honest open fill needs open-to-close returns
threaded through the entry bar, which cannot be done as arithmetic on a
position series, and doing it properly would mean precomputing a second return
series for every configuration. A one-bar execution delay is the standard
vectorised treatment and it is a real cost of being late, but it is not the
same thing, so the page states the difference rather than overclaiming.

**The guard.** Putting the backtest in JavaScript means the same arithmetic now
exists twice, in two languages, with nothing stopping the two from drifting.
`scripts/check_site_data.py` ports the JavaScript back into Python, runs it
against `site/data.json`, and compares against `src/backtest.py` and
`src/metrics.py` computed from scratch -- five tickers, two models, four
schemes, two cost levels. It also checks that the encoding round-trips, that
every series the interface can request exists, and that no bar on or after
2023-01-01 is in the file. It passes, with the largest disagreement at 2e-06,
which is the six-decimal rounding. It exits non-zero on any mismatch.

That script is the part of step 11 the work split actually assigns to me --
"Vansh reviews the data contract" -- and a review that is a script rather than a
reading is one that still works in a fortnight.

**`site/data.json` is committed.** The gitignore says the precomputed grid gets
committed deliberately when it is final rather than incidentally while
iterating. It is not final -- the floor question is open -- but without it the
site cannot be demonstrated from a clone, and 475 KB is not worth the
inconvenience. Regenerable with one command, and `check_site_data.py` will fail
loudly if it ever goes stale against the modules.

**One curiosity worth a sentence in the write-up.** Logistic's Sharpe
*improves* when execution is delayed by a bar, from about -0.04 to +0.17 at
10 bps on SPY. A signal that does better when you act on it late is a signal
that is anti-predictive at the one-day horizon, which is the same thing the
negative selection edge said on 2026-11-14 from a different direction.

**Caveat on who did this.** The work split assigns steps 11 and 12 to Aksh, and
the open questions reserved the site stack for him specifically because he was
building it. I built it at Vansh's instruction while Aksh was unavailable. The
stack choice -- plain HTML, no framework, no charting library -- is therefore
mine rather than his, and it is deliberately the easiest possible thing to
replace: one file, no build step. If Aksh wants a framework the data contract
does not change, and `check_site_data.py` will keep whatever replaces it honest.

### 2026-11-15 · STEP 13: THE HOLDOUT IS OPEN. This is the result.

Opened (Vansh and Aksh, both present, once) by `scripts/open_holdout.py`.
2023-01-01 to 2025-12-31, declared in `config.yaml` on 2026-10-14 before any
modelling. Each model fitted a single time on the purged pre-holdout rows --
about 4,300 per ticker -- and predicted roughly 744 holdout bars. Nothing was
tuned, selected or retried. Written down here before being discussed, per the
working agreement.

**Classification, averaged over the five ETFs:**

| model | AUC | vs floor | precision | positive rate | coverage |
| --- | --- | --- | --- | --- | --- |
| buy_and_hold | 0.5000 | -0.044 | 0.437 | 0.437 | 1.000 |
| ma_crossover | 0.5014 | -0.043 | 0.438 | 0.437 | 0.675 |
| logistic | **0.5754** | +0.031 | 0.503 | 0.437 | 0.043 |
| gbt | 0.5318 | -0.012 | 0.481 | 0.437 | 0.190 |
| meta_crossover_logistic | 0.5392 | -0.005 | 0.508 | 0.437 | 0.132 |

**Strategy at 0 bps:**

| model | Sharpe | s.e. | t | max DD | turnover | exposure | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| buy_and_hold | **1.116** | 0.580 | 1.92 | 0.206 | 0.5 | 1.00 | +81% |
| ma_crossover | 0.948 | 0.580 | 1.63 | 0.135 | 4.6 | 0.72 | +47% |
| logistic | 0.579 | 0.580 | 1.00 | 0.104 | 7.7 | 0.12 | +15% |
| gbt | 1.048 | 0.580 | 1.81 | 0.102 | 18.6 | 0.42 | +43% |
| meta_crossover_logistic | 0.695 | 0.580 | 1.20 | 0.079 | 8.2 | 0.23 | +20% |

At 10 bps: buy-and-hold 1.113, trees 0.879, logistic 0.470.

**Deflated, and exposure-matched, at 0 bps:**

| model | Sharpe | matched benchmark | selection edge | t (n=5) | DSR |
| --- | --- | --- | --- | --- | --- |
| buy_and_hold | 1.116 | 1.122 | -0.006 | -0.6 | 0.817 |
| ma_crossover | 0.948 | 0.975 | -0.028 | -0.3 | 0.792 |
| logistic | 0.579 | 0.400 | +0.179 | 1.1 | 0.631 |
| gbt | 1.048 | 0.756 | **+0.292** | 3.4 | 0.860 |
| meta_crossover_logistic | 0.695 | 0.576 | +0.120 | 0.7 | 0.674 |

---

**What it says.**

**1. The AUC held, to a degree nobody expected.** Logistic scored 0.5755
under the purged scheme pre-holdout and **0.5754** on the holdout -- a
difference of one ten-thousandth, on data from three years nobody had looked
at. Every model held within 0.024. Whatever the pipeline was measuring, it was
measuring it stably, and the purged estimate was an honest predictor of
out-of-sample performance. That is the single best piece of evidence that the
evaluation machinery works, and it is what the whole project was built to
establish.

**2. Buy-and-hold still wins, and by more than before.** Its Sharpe went from
0.480 to 1.116. That is not skill appearing, it is the market: SPY +88%,
QQQ +138%, GLD +133%, IWM +49%, TLT -3% over the period. Every strategy's
Sharpe roughly doubled for the same reason. No model beat the index, which is
the same conclusion as pre-holdout.

**3. Nothing is statistically distinguishable.** The Sharpe standard error is
**0.580**, against 0.242 pre-holdout, because the holdout is 744 bars rather
than 4,300. Not one model reaches t = 2. The holdout is too short to resolve
any of this, which was knowable in advance from the sample-size arithmetic and
is a limitation of the design rather than a surprise.

**4. The selection edge flipped sign, which tells us it was noise.** Logistic's
edge was **-0.164** pre-holdout at t = -2.36, the one result that cleared two
standard errors, and we wrote that it was "genuinely, measurably picking bad
days". On the holdout the same model's edge is **+0.179**. A quantity that
reverses sign out of sample was noise, and the pre-holdout significance was a
false positive -- exactly the kind this project exists to warn about. It is
uncomfortable that we found it in our own strongest negative claim, and it goes
in the write-up for that reason.

**5. The trees' +0.292 edge at t = 3.4 is not a discovery, and we are not
reporting it as one.** It is one number among about sixty in this table, drawn
from five tickers with four degrees of freedom, and its pre-holdout value was
+0.071 at t = 1.37 -- it quadrupled, which is what noise does. The Deflated
Sharpe Ratio exists precisely to price this kind of thing and puts the trees at
**0.860**, short of any convincing bar. Reporting it as skill would be the
error the entire project is about. If anything it is a useful closing
illustration: even after all this machinery, the single most impressive-looking
number in the final table is one we have to decline.

**Honest summary for the write-up.** The purged estimate predicted
out-of-sample AUC almost exactly. The strategies' profitability was dominated
by a strong bull market. No model beat buying the index, nothing survives a
multiple-testing correction, and our one significant pre-holdout finding
reversed. The project's claim is about the measurement machinery, and the
machinery was vindicated; the models were not.

**The holdout is now closed permanently.** Nothing upstream of this entry may
be changed on the strength of it. No barrier width, feature, model setting,
embargo or threshold moves from here.

### 2026-11-15 · Reversal: the AI-use disclosure IS required

Vansh reversed this after reading the submission instructions. On 2026-11-10 we
removed the AI-disclosure rule from the block at the top of this file, on the
strength of a recollection that the professor had said it was not needed. The
submission instructions state plainly that the repository must contain "your
AI-use disclosure and citations".

So the rule goes back, and the disclosure now exists as a section of the
README rather than scattered through this log: what AI assistance was used for,
and that the design decisions and their reasoning are ours and recorded here.

Worth recording why this happened, because it is the same failure mode the
project is about. A claim about the course requirements was asserted without
being checked against the written instructions, acted on, and then a rule was
deleted on the strength of it. The instructions were sitting in `../planning/`
the whole time. Neither of us read them until step 13 was due.

### 2026-11-15 · The website is the submission, so it now carries the argument

Vansh changed this after reading the submission instructions. Our site was a
control panel over the backtest. The instructions say a grader will assess the
project "entirely from what you submit, including your understanding, which we
will judge from how clearly the website reasons from the question to the
conclusions", and list eight things it must cover.

`docs/index.html` is now the full report: the question and what one row is; the
data with its source, size and the dividend-adjustment problem; the label and
the barrier-width decision that nearly broke the project; preprocessing and the
in-fold leak; the baseline and the five models; the three schemes, the metrics
and their baselines including the 0.544 floor; the results with the gap as a
range; the error analysis; the interactive panel; the holdout; and eleven
limitations. The panel is the required interaction and it is kept, because
"switch the scheme and watch the curve not move" is the finding made visible.

Two choices behind it.

**Every table renders from `docs/data.json`, not from hard-coded HTML.**
`scripts/build_site_findings.py` reads the result CSVs and writes a `findings`
block beside the position series. On the graded page, a transcribed number is a
number that will eventually be wrong; this way rerunning an experiment and
rebuilding updates the page with no edit. Verified: every table id in the HTML
has a renderer and data behind it, and the ten headline figures the prose
asserts match the JSON exactly.

**`site/` is now `docs/`.** GitHub Pages serves a repository's `docs/` folder on
the default branch with one setting and no CI workflow. One setting is the least
that can break the day before a deadline.

### 2026-11-15 · Results regenerated after the review fixes

Housekeeping (Vansh). `outputs/step06_*.csv` and `step789_*.csv` were still from
2026-11-14, before the review fixes to pooled precision, the positive-rate
benchmark and the paired standard error. The website presents those numbers, so
both scripts were rerun and the site rebuilt from the new CSVs. The code had
been correct since the fixes; only the saved tables were stale.

### 2026-11-15 · Attribution wording levelled out

Vansh changed this. Several early entries attributed suggestions to the AI
assistant by name — "Claude flagged", "Claude proposed". Those now read
"flagged in review" and "proposed in review", and the two session-level notes
are shorter. Nothing about what was decided, or by whom, has changed: every
entry still says whether we accepted a suggestion or made the call ourselves.
The AI-use disclosure lives in the README, which is where the submission
instructions ask for it.

Recorded because this file's own rule is not to edit old entries, and
rephrasing attribution across ten of them is an edit even though no decision
moved.

### 2026-11-17 · Would narrower barriers have scored better? No, the opposite

Vansh asked whether setting the barriers lower would have given us a better
AUC, and why we had not tried it. We measured it instead of guessing:
`scripts/check_barrier_sensitivity.py`, logistic under the purged scheme,
pre-holdout, five funds.

The thing that makes this answerable at all is that **raw AUC is not
comparable across barrier widths**. The no-signal floor moves with the width:
wide barriers time out more often, a timeout counts as label 0, and trailing
volatility predicts timeouts, so a model with no skill whatsoever scores higher
when the barriers are wider. Comparing AUC at 1.0 against AUC at 2.5 would
mostly be comparing two different floors. So we measured the floor at every
width too, from the sign-flip null, and compared the gap.

| width | label 1 | timeout | window | effective N | AUC | floor | lift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0σ | 0.534 | 0.3% | 1.9 bars | 1,575 | 0.5049 | 0.5024 | +0.0025 |
| 1.5σ | 0.520 | 4.0% | 3.2 bars | 1,128 | 0.5130 | 0.5059 | +0.0071 |
| 2.0σ | 0.474 | 14.0% | 4.7 bars | 831 | 0.5401 | 0.5108 | +0.0293 |
| **2.5σ** | 0.405 | 27.4% | 6.0 bars | 662 | 0.5755 | 0.5268 | **+0.0487** |
| 3.0σ | 0.329 | 41.0% | 7.0 bars | 563 | 0.6030 | 0.5424 | +0.0606 |

**Narrower would have been worse, not better.** At 1.0σ the model scores 0.5049
against a floor of 0.5024 — a lift of two and a half thousandths, which is
nothing. The label at that width is close to "which way did tomorrow move", and
one-day direction is not predictable from anything we have. Our intuition that
a tighter target is an easier question had it backwards: a tighter target is a
*noisier* question.

AUC, the floor and the lift all rise monotonically with width. The floor
confirms the mechanism we found on 2026-11-10 from the other direction: it
falls from 0.527 at 2.5σ to 0.502 at 1.0σ, because there are almost no timeouts
left for volatility to predict. It also cross-checks our recorded figure — this
run puts the 2.5σ floor at 0.5268 against the 0.529 we measured with twice as
many seeds.

**We are not changing the width, and could not.** 3.0σ shows a slightly larger
lift, +0.061 against +0.049, for 100 fewer effective observations. The holdout
was opened on 2026-11-15 and our rule is that nothing upstream of it moves on
the strength of anything measured afterwards. Switching now would mean picking
a configuration after seeing holdout results, which is the exact failure this
project is about.

It does mean the 2.5σ we chose on 2026-11-01, for reasons that had nothing to
do with accuracy, happens to sit near the top of the achievable lift without
paying 3.0σ's cost in sample size. That is luck rather than judgement and the
write-up should say so.

**Trial counter: unchanged, at 5.** Same reasoning as the embargo sweep on
2026-11-15. This measures how stable one fixed configuration is, and we report
the range rather than selecting from it — which here is not a matter of
discipline but of the holdout already being open. If anyone ever does select a
width on these numbers, the counter moves by 5 and the holdout result is void.

The lift/sd column the script prints is a rough yardstick rather than a
significance test, since it compares the lift against the spread of the null
runs and ignores the AUC's own fold-to-fold error. Read loosely, none of these
lifts is convincingly significant, which is consistent with everything else we
have found.

### 2026-11-17 · An outside review found four bugs. Three were real

Vansh had another model review the repository. It reported four bugs. We
checked each against the code rather than taking them on trust, and fixed the
three that held up.

**Real: `tests/test_synthetic.py` assumed the working directory.** Line 102
read `data/raw/SPY_...csv` as a bare relative path while every other test
resolves from `ROOT`. Running `pytest` from the parent folder failed with a
`FileNotFoundError`. Our own inconsistency, and a grader running pytest from
anywhere but the repository root would have hit it. Fixed; 161 tests now pass
from either directory.

**Real, and worse than reported: `build_site_data.py` wiped the results
tables.** It rebuilds `docs/data.json` from scratch, so running it discarded
all 16 `findings` sections that `build_site_findings.py` writes. Confirmed by
running it and counting the keys: 16 before, none after.

The review said this leaves the tables blank. It is worse than that. The page
calls `table("t-data", ...)` without checking the section exists, so a missing
`findings` block throws inside `renderTables`, the exception escapes into the
`fetch` chain, and the catch handler replaces the whole page with the "could
not load data.json" error. The interactive panels would have gone too.

Fixed on both sides: `build_site_data.py` now carries any existing findings
over, and every table on the page is guarded so a section that has not been
generated leaves one table hidden rather than taking the page down.

**Real, with a fix we did not take: a fresh clone crashed with a traceback.**
`data/processed/` is gitignored because it is regenerable, so
`check_site_data.py` on a fresh clone died inside pandas with a
`FileNotFoundError`. The review suggested running `data_process.py` and
`build_labels.py` automatically through `subprocess`. We did not: a script
whose job is to verify the data should not silently regenerate it. It now
checks first and prints the two commands to run.

**Overstated: the GBT position mismatch.** The review reported
`check_site_data.py` failing on `GLD/gbt` at embargo 0 and 20, blaming
predicted probabilities "around 0.499 to 0.519" being flipped by
floating-point differences across scikit-learn builds. The upper figure is
wrong -- 0.519 is nowhere near the 0.5 boundary and no rounding difference
moves it. Measured on GLD, 14 of 4,303 predictions sit within 1e-3 of the
boundary and 2 within 1e-4.

But the conclusion survives the bad arithmetic. `HistGradientBoosting` sums
leaf values in an order that depends on the OpenMP thread count, so a
prediction that close to the boundary can genuinely land on either side on
another machine, and each flip moves about six bars of the position series.
Fourteen borderline predictions is more than enough to produce the ~56
differing bars they saw. The check passes on our machine and we could not
reproduce the failure, which is exactly what an environment-dependent bug
looks like.

We did not take the suggested fix either, which was to allow a 2% mismatch
everywhere. That is 86 bars of slack applied to models where any difference at
all is a bug. Instead the check now splits by whether a model is fitted:

- `buy_and_hold` and `ma_crossover` are not fitted, so any difference remains a
  hard failure.
- The three learned models may differ on up to 1% of bars, which is reported as
  drift rather than failure -- and the statistics still have to agree to 1e-4,
  so anything that actually changed the result still fails.

Checked that this still catches a real break: corrupting `GLD/gbt` by 8% of
bars fails as it should.

The review's second recommendation, that `docs/data.json` stay committed with
all 16 findings keys, follows from the second fix and now holds.

### 2026-11-17 · Aksh's audit, and the two sections we had already lost

Vansh reviewed `planning/AUDIT_AND_ACTION_PLAN.md`, written by Aksh with an
assistant. Its Section 1 repeats the four bugs from the PDF review, which we
had already dealt with. Its Section 2 proposes a frontend rebuild and says an
enhanced version "has been built and tested" at `docs/enhanced.html`. **That
file does not exist in the repository.** Nothing was lost, but the plan
describes work that was not delivered, and we should not have taken it on
trust.

**The important finding was ours, not theirs.** Checking the site against the
submission instructions line by line showed it was missing two of the eight
required items. There was no description of our preprocessing choices -- no
mention of the one-bar shift, the in-fold scaler, or why `sigma` is withheld --
and no section on the baseline and the models at all. Meta-labelling appeared
nowhere on the page.

Both were cut on 2026-11-15 when the site was trimmed for being too long. That
trim went too far and took required content with it. The page now has §5
Preprocessing and §6 The models, and every one of the eight requirements is
checked by a script rather than by eye.

**What we took from the frontend proposal.** A sticky section rail with
scroll-spy, because thirteen sections is a lot to navigate and a grader needs
to find the eight required ones. A crosshair and inspection tooltip on the
equity chart, which turns a picture into something you can read values off.
Styled range sliders, since the native ones look unfinished. A reproduce block
with copy-to-clipboard, which shortens the path from the site to checking it.

**What we turned down, and why.**

*The dark/light toggle.* Our identity is warm oat paper. A dark variant of that
is not the same design, it is a second design, and maintaining two would double
the surface for three canvases and the contrast work we just did. A theme
switcher is also one of the most common signals that a page was generated
rather than designed. One committed light theme, deliberately.

*A separate reading-progress bar.* The scroll-spy rail already says where you
are and, unlike a bar, lets you jump. Two indicators for one job.

*Their Fix 2, auto-running the data build through `subprocess`.* A script whose
job is to verify the data should not quietly regenerate it. Ours names the two
commands instead.

*Their Fix 3, a flat 2% position tolerance for `gbt` only.* Two problems. 2% of
4,312 bars is 86 bars of slack. And it exempts the wrong model: logistic has
*more* predictions near the 0.5 boundary than the trees do, 17 against 14 on
GLD, so their rule would still fail across machines on the model they left
strict. Ours splits on whether a model is fitted at all, allows 1%, reports it,
and still requires the statistics to match.

Contrast was re-checked across all fifteen foreground and background pairs
after the rebuild. No failures.

---
## Open questions

Move these into the log as dated entries once decided.

- **JSON size for the website.** Precomputing all 2 × 21 × 21 × 2 = 1,764
  slider combinations as full equity curves is hundreds of megabytes. Plan:
  ship the ~42 position series (purging × embargo) and apply cost and
  execution lag in the browser. Confirm before building.
- **Adopt the sign-flip floor of 0.529 in place of the synthetic 0.544?**
  Needs Aksh. It is better calibrated and per ticker, and it raises
  logistic's lift from +0.032 to +0.047. `scripts/check_floor_real.py`
  prints it; wiring it in is a one-line change to `NO_SIGNAL_FLOOR`.
- **Is the DSR a live number or a fixed one?** It depends on the total trial
  count, which is a property of the whole project, not of a slider position.
  Probably display it fixed, with the per-configuration Sharpe live.
- **Final feature list.** Currently about eight. They only enter the trial
  counter once feature sets are scored against each other at step 6.
- **Split of work between Vansh and Aksh.** Both must be able to derive
  purging and the DSR — the viva asks each of us separately. Do not split it
  as "one codes, one does the website".
