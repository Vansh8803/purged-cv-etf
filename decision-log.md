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
| Barrier width | ±1 × 20-day realized volatility | 2026-10-14 |
| Holdout | 2023-01-01 to 2025-12-31, UNOPENED | 2026-10-14 |
| Primary CV | Purged 5-fold, embargo ≈1% of sample | 2026-10-14 |
| Control CV | Standard 5-fold K-fold | 2026-10-14 |
| Third scheme | Walk-forward, expanding window | 2026-10-14 |
| Repo | `purged-cv-etf` | 2026-10-21 |

## Trial counter

The DSR needs the number of configurations actually tested. Increment this
every time a new combination is evaluated — feature set, barrier width,
horizon, embargo length, model type, hyperparameter grid point. Undercounting
makes the headline number worthless.

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
- **Final feature list.** Currently about eight. Each one added is a new
  configuration for the trial counter.
- **Split of work between Vansh and Aksh.** Both must be able to derive
  purging and the DSR — the viva asks each of us separately. Do not split it
  as "one codes, one does the website".
