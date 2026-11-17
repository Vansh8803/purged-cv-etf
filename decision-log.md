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
disclosure lives in the README.

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

We went this way because the methodology is the part we can actually settle.
Chasing prediction would have meant reporting an accuracy this data cannot
deliver, while the question of how much validation choices move a reported
score is answerable with the data we have.

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

A single gap number says nothing about which models we compared or where they
fail, and it contains no errors to show. Three results cover what one number
cannot.

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

It is about fifty lines of code, and importing the one thing our project is
about would be absurd.

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

At 1.0 the barriers were narrower than a typical day's high-to-low range, so
almost everything touched immediately: the median holding period was one bar
and the ten-day limit bound on under 1% of rows. Windows that short barely
overlap, so purging would have had nothing to remove and our headline number
would have been an artifact of the barrier width rather than anything about
cross-validation.

We compared 2.5 against sqrt(10) = 3.162, the width at which a barrier matches
a typical ten-day move, and took 2.5 because it keeps about 20% more effective
sample and because at 3.162 nearly half of all labels are timeouts.

The cost is real and belongs in the limitations: effective sample size falls
from roughly 1,575 to 662. Wider barriers mean longer windows, more overlap and
less independent data. That is the correct trade, since the overlap is what we
are studying, but it is a trade.

### 2026-11-01 · A same-bar tie counts as label 1

When the high clears the upper barrier and the low breaks the lower barrier on
the same bar, we label it 1.

Daily bars cannot tell us which level was touched first, so this is a choice
rather than a measurement. We take 1 because the upper barrier is the side we
would have acted on. Calling every ambiguous bar 0 is more conservative, but it
throws away exactly the volatile bars where any signal would live. Worth
revisiting with intraday data, which we do not have.

### 2026-11-01 · t1 counts trading bars, and unresolved rows are dropped

Two decisions about how a label resolves.

**t1 is a position in the trading calendar, not a calendar offset.** Adding ten
calendar days to a Thursday gives a Sunday, which is not a bar in any of our
data. Purging is defined as an overlap test on `[t, t1]`, so a t1 that is not a
real bar makes every purge decision slightly wrong.

**A row whose window runs past the end of the data is dropped, not labelled.**
Shifting the future off the end gives missing values, missing values compare as
"no touch", and those rows were being recorded as timeouts. They had not timed
out, they simply had no future to look at.

### 2026-11-01 · Barriers are computed in log-return space

`sigma_t` is the standard deviation of daily *log* returns, so we place the
barriers as a log move off the close:

    upper = close_t * exp(+2.5 * sigma_t)
    lower = close_t * exp(-2.5 * sigma_t)

`project_details.md` said "cumulative return from the close of t", which could
also have been read as `close * (1 +/- 2.5 * sigma)`. In simple-return space
the up move and the down move are different sizes in log terms, so the label
would inherit a small upward bias that has nothing to do with the market. Log
space makes the two barriers exact mirrors: `upper * lower == close^2`.

### 2026-11-01 · We report effective sample size, not row count

Overlapping labels are not independent observations, so we report average
uniqueness times the row count beside every raw count.

Measured with `src/uniqueness.py`: average uniqueness 0.154, so our 4,302
pre-holdout rows per fund carry about **662 independent observations**. Every
error bar in the project follows from that number, and it is why we refused a
hyperparameter search and held the gradient-boosted trees to a small capacity.

### 2026-11-03 · A test block's reach is its maximum t1

Aksh decided how we measure a fold's span. When PurgedKFold purges around a
test fold, that span is `[first test row's t, the largest t1 among the test
rows]`, not `[first test row's t, last test row's t]`.

A test label resolves up to `horizon` bars after its own row, so the stretch of
history a fold occupies runs past its last row. A training row starting in that
tail overlaps a test label and would survive a purge measured by row dates
alone. `src/purging.py` exposes this as `block_span()` rather than leaving it
inline, because the walk-forward splitter faces the identical choice and the
two must not diverge.

### 2026-11-03 · Embargo bounds, and what sets its length

Aksh settled the embargo's bounds. It drops training rows starting in
`(test_end, test_end + embargo]` — exclusive at the start, inclusive at the
end, counted in trading bars rather than calendar days, and clipped at the end
of the sample rather than running off the index.

Exclusive at the start because `test_end` is already purged by the overlap
rule, so including it again would be double counting. Inclusive at the end so
that an embargo of 1 actually withholds one bar.

The usual reason given for an embargo is serial correlation, which is true but
vague. Ours is concrete: our features include 20-day rolling windows, so a
training row just after a test fold has features computed from bars inside it.
That also gives the length a floor of at least the longest feature lookback.
`embargo_frac` of 0.01 gives 43 bars, comfortably above the 20 the rolling
features need. The 200-day moving average cannot be covered by any embargo we
could afford, and belongs in the limitations.

### 2026-11-10 · The control is contiguous K-fold

Our control is **standard 5-fold K-fold with contiguous folds**, and it is the
only control our results tables carry.

`project_details.md` says "standard 5-fold K-fold", and `sklearn.KFold` does
not shuffle by default, so that phrase already means contiguous. It is also the
harder test for our own claim: a shuffled control leaks 88% of training rows
against 0.33% contiguous, and would have handed us a large gap against a
control nobody sensible would use.

What this costs us, stated plainly: **the headline gap will be small.** Purging
removes 0.33% of training rows from a contiguous control, so the difference in
score will be modest and may well be inside noise. That is the honest answer
and it is the one we report.

The shuffled figure is kept as one sentence of context.
`NaiveKFold(shuffle=True, seed=...)` exists so that sentence is reproducible.
It does not get its own row in the model comparison or its own gap number.

### 2026-11-10 · Walk-forward keeps an embargo that does nothing

`WalkForward` accepts an embargo and ignores it, and we kept the parameter
anyway so the splitters stay interchangeable.

The embargo drops training rows starting after the test block, and an expanding
window has none by construction. The consequence lands in the write-up rather
than the code: walk-forward is flat across the 0-to-20 embargo sweep, and a
flat line next to a sloping one needs a sentence rather than a fix.

Purging still applies to it. Training only on the past does not stop a label
from resolving into the test block, and on the toy case the purge removes
exactly those boundary rows.

### 2026-11-10 · Preprocessing is fitted inside folds, and sigma is withheld

The features stay globally computed; the preprocessing does not.

Every feature is a deterministic rolling function of its own past, shifted one
bar, so a row's values do not change with the fold it lands in. A scaler is
different: fitted on the whole matrix, every fold's test rows contribute to the
numbers used to transform them. `make_pipeline` therefore returns an sklearn
Pipeline, so the only way to use it is to fit inside a fold.

`FEATURES` is an explicit list of ten and `EXCLUDED` spells out what is
withheld, because "everything except label and t1" would quietly admit the raw
bars. Two exclusions are worth naming. `log_return` is the return *at* t,
unshifted; `ret_1d` is the same number shifted and is the one that belongs.
`sigma` is sigma_t, unshifted, and it *sets the barrier width* — handing it
over would tell the model how far price must travel for the label to fire.

No feature selection. Ten features against an effective sample near 660 does
not need it, and a selector is another thing to fit in-fold.

### 2026-11-10 · Real-data AUC is read against 0.544, not 0.500

We report every AUC against a no-signal floor of 0.544 rather than against
chance.

On synthetic series with no price predictability at all, the full feature set
still scores about 0.544 while the return features alone score 0.503. The cause
is our own label: `sigma_t` sets the barrier width, wide barriers are less
likely to be touched, an untouched barrier times out, and a timeout counts as
0. Trailing volatility therefore predicts the label with no directional skill.
Of the rows that resolved by touching a barrier, 50.9% went up.

We are not redesigning the label. The binary target was decided on 2026-10-14
for reasons that still hold, and a three-class target would break comparability
across the five models. What changes is how the number is read.

We also average every synthetic figure over ten seeds, because single runs at
phi = 0 landed anywhere between 0.489 and 0.623.

### 2026-11-10 · One number per question, not a matrix of options

Where we have made a choice and recorded it, the write-up reports the branch we
chose and mentions the alternative in a sentence. It does not report both as
parallel results.

Several decisions had drifted towards "run both and report the pair" — two
controls, a 21-point embargo sweep, cost and lag crossed on the site. Each is
cheap on its own and together they multiply into a table nobody reads. A
project with one clear claim beats one with six hedged ones.

This does not touch steps 1 and 3, the synthetic check, or the holdout
discipline. This log keeps every alternative we considered regardless of what
the write-up carries: the log is the audit trail, the write-up is the argument.

### 2026-11-14 · How we compute precision, its benchmark, and the gap's error

Three measurement decisions, all of which changed published numbers.

**Precision is pooled across folds, not averaged.** Our models act on as little
as 11% of bars and the per-fold action count swings tenfold inside one ticker.
Averaging ratios gives a 22-action fold the same weight as a 225-action one,
and a small denominator reliably reads high.

**Precision is read against the positive rate, not the majority rate.** At 40%
positives those are 0.40 and 0.60. The positive rate is what acting every day
scores, so it is the number to beat. Coverage travels with it, because
precision alone rewards abstention.

**The control-versus-purged gap gets a paired standard error.** Both schemes
split the same rows the same way, so their test folds are identical and only
the training sets differ. The noise on the gap is the spread of the per-fold
differences, about ten times tighter than either scheme's own error.

### 2026-11-14 · The backtest holds a position for the label's own life

Two decisions in `src/backtest.py`.

**The holding period comes from the label.** A signal at the close of *t* opens
a position on the next bar and holds until `t1`. The model was trained to
answer "does the upper barrier fall first", so the backtest runs that trade. A
fixed ten-day hold would measure something the model was never asked.

**Overlapping signals do not stack.** Position is capped at one unit. Labels
overlap by construction, so stacking would make leverage a function of signal
density and turnover meaningless. The cost is that a run of signals looks like
a single one.

### 2026-11-14 · We report selection edge with exposure held constant

Alongside Sharpe we report an exposure-matched benchmark: the Sharpe from
holding the same *number* of bars, chosen at random, averaged over 200 shuffles.

A long-or-flat strategy invested 22% of the time forfeits most of the market's
drift whatever it selects, so comparing its Sharpe against buy-and-hold
confounds how well it chose with how often it was in. Subtracting the matched
benchmark separates the two. Buy-and-hold validates the control: at ~100%
exposure there is nothing to shuffle and its edge comes out at -0.001.

### 2026-11-14 · Trial counter is 5

The counter is **5** — the five models in `model_zoo`, scored per ticker.

A trial is a configuration we got a performance number out of and could
therefore have selected on. The three schemes are measurements of the same
model rather than alternatives we choose between, so they do not multiply it.
Each ticker gets its own model, so the five tickers do not multiply it either.
No hyperparameter grid was run.

Sweeps do not move it. A sensitivity analysis over one parameter — the embargo
from 0 to 20 bars, or the barrier width — measures how stable one configuration
is, and we report the range rather than picking from it. If anyone ever selects
a setting on those numbers, the counter moves and this entry is wrong.

### 2026-11-15 · The sign-flip floor is measured but not adopted

We measured the no-signal floor a second way, on real bars, by randomising the
sign of each daily return. That keeps the volatility path intact — rolling
sigma correlates 0.998 with the real one — while destroying all directional
information. It gives 0.529 against the synthetic 0.544.

We have not adopted it. The synthetic figure is conservative, so nothing we
reported was flattered by the difference, and switching the headline floor
needs both of us to agree. `scripts/check_floor_real.py` prints it; the open
question below records that it is unresolved.

### 2026-11-15 · The site ships position series, not equity curves

Crossing all four site controls is 2 x 21 x 21 x 2 = 1,764 combinations, and a
full equity curve for each is hundreds of megabytes. We ship the position
series instead, run-length encoded, and the browser applies cost and execution
lag arithmetically. The whole grid is 475 KB.

**"One bar late" is a one-bar execution delay, not a next-open fill.** An
honest open fill needs open-to-close returns threaded through the entry bar,
which cannot be done as arithmetic on a position series. A delay is a real cost
of being late, but it is not the same thing, and the page says so.

Putting the backtest in JavaScript means the same arithmetic now exists twice.
`scripts/check_site_data.py` ports it back into Python and fails if the two
disagree.

### 2026-11-15 · The holdout was opened

Opened by Vansh and Aksh, both present, once, by `scripts/open_holdout.py`.
2023-01-01 to 2025-12-31, declared in `config.yaml` on 2026-10-14 before any
modelling. Each model was fitted a single time on the purged pre-holdout rows
and predicted roughly 744 bars. Nothing was tuned, selected or retried, and the
numbers were written down before we discussed them.

The full results are in `outputs/step13_holdout.csv`. The one line worth
keeping here: logistic scored 0.5755 under the purged scheme beforehand and
0.5754 on the holdout, and every model held within 0.024.

**The holdout is now closed permanently.** Nothing upstream of this entry may
change on the strength of it — no barrier width, feature, model setting,
embargo or threshold.

### 2026-11-17 · The barrier width stays at 2.5

We measured what the barrier width cost us, after the fact, and changed
nothing.

Raw AUC is not comparable across widths, because the no-signal floor moves with
the width. Measuring both at each width, the lift over the floor is +0.002 at
1.0 sigma, +0.049 at our 2.5, and +0.061 at 3.0. Narrower would have been
worse, not better: at 1.0 the label collapses towards "which way did tomorrow
move", which is not predictable from anything we have.

3.0 shows a slightly larger lift for a hundred fewer effective observations. We
are not switching, and could not: the holdout is open, and changing a setting
now would mean choosing a configuration after seeing holdout results.

It does mean the 2.5 we chose for unrelated reasons sits near the top of the
achievable lift without paying 3.0's cost in sample size. That is luck rather
than judgement and the write-up should say so.

---
## Open questions

Move these into the log as dated entries once decided.

- **Adopt the sign-flip floor of 0.529 in place of the synthetic 0.544?**
  Needs Aksh. It is better calibrated and per ticker, and it raises
  logistic's lift from +0.032 to +0.047. `scripts/check_floor_real.py`
  prints it; wiring it in is a one-line change to `NO_SIGNAL_FLOOR`.
- **Is the DSR a live number or a fixed one?** It depends on the total trial
  count, which is a property of the whole project, not of a slider position.
  Probably display it fixed, with the per-configuration Sharpe live.
