"""Step 6: the models, and the rules they have to beat.

All five predict the same binary target -- did the upper barrier fall first --
so precision and AUC mean the same thing in every row of the comparison. That
was settled on 2026-10-14, when the draft had some models predicting three
classes and some two, which makes a side-by-side table meaningless.

Two of the five are not fitted at all. A rule dressed as a classifier is scored
through exactly the same code path as a learned one, which is the only way the
comparison is like for like: same folds, same test rows, same metric, same
purging. `buy_and_hold` acts every day and is the "never skip" reference;
`ma_crossover` is the baseline from the proposal, and beating it or failing to
is informative either way.

**Scaling and the rules.** The learned models sit behind `make_pipeline`, which
standardises each column independently. The crossover rule compares two
different columns to each other, and standardising them separately does not
preserve that comparison -- each gets its own mean and standard deviation. So
the rules read the raw design matrix and are deliberately not wrapped in the
pipeline. They need no imputation or scaling: they are a comparison of two
numbers that are either both present or both absent.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src.features import make_pipeline


def crossover_signal(X):
    """True where the 20-day moving average sits above the 50-day one.

    Read off the distance-from-MA features rather than recomputed from price,
    so the rule inherits the same one-bar shift as every other feature and
    cannot act on a number before it existed. The proposal's "applied with a
    one-bar lag" is therefore structural here rather than something to remember.

    The algebra, because it reads backwards. With `d20 = (C - MA20) / MA20`,
    we have `MA20 = C / (1 + d20)`, and likewise for 50. Price is positive and
    so is every moving average, so `1 + d` is positive and

        MA20 > MA50  <=>  C / (1 + d20) > C / (1 + d50)  <=>  d20 < d50.

    The inequality flips. A rule written the intuitive way round would be the
    exact opposite of a golden cross, which is the kind of error that still
    produces a plausible-looking score.
    """
    return (X["dist_ma_20d"] < X["dist_ma_50d"]).to_numpy()


class _Rule(BaseEstimator, ClassifierMixin):
    """A deterministic rule with the estimator interface, so it is scored
    through the same `cross_val_scores` as everything else.

    `fit` ignores its training data on purpose. That is not laziness: it is what
    makes these the reference points. A rule cannot leak, cannot overfit, and
    scores identically under every scheme, so any difference a learned model
    shows between the control and the purged run is the learning, not the data.
    """

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        return self._signal(X).astype(int)

    def predict_proba(self, X):
        p = self._signal(X).astype(float)
        return np.column_stack([1.0 - p, p])


class AlwaysAct(_Rule):
    """Buy and hold, expressed as a classifier: act on every bar.

    Its precision is the positive rate by construction, which is exactly the
    number every other model's precision has to be read against. Its AUC is 0.5
    because it never discriminates.
    """

    def _signal(self, X):
        return np.ones(len(X), dtype=bool)


class MACrossover(_Rule):
    """The baseline: act while the 20-day MA is above the 50-day, lagged a bar."""

    def _signal(self, X):
        return crossover_signal(X)


class MetaLabel(BaseEstimator, ClassifierMixin):
    """The crossover picks the direction; the model only decides whether to act.

    Asking which way price will move is close to hopeless on daily bars. Asking
    whether to take a signal somebody else generated is a binary question on a
    smaller, more homogeneous sample, and precision on it means something.

    Trained only on the rows where the rule fires, and predicts 0 everywhere
    else, so it can never act where the rule would not have. That is the point
    of the construction and the test suite pins it.

    A fold where the rule never fires, or fires only on one class, leaves
    nothing to learn from. Rather than crash or quietly fit a degenerate model,
    it falls back to predicting 0 and records that in `fitted_`.
    """

    def __init__(self, inner=None, min_train=30):
        self.inner = inner
        self.min_train = min_train

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        rows = np.flatnonzero(crossover_signal(X))
        y_sub = y.iloc[rows]

        self.fitted_ = len(rows) >= self.min_train and y_sub.nunique() > 1
        if self.fitted_:
            self.inner_ = clone(self.inner)
            self.inner_.fit(X.iloc[rows], y_sub)
        return self

    def _act(self, X):
        sig = crossover_signal(X)
        if not self.fitted_:
            return sig, None
        rows = np.flatnonzero(sig)
        return sig, rows

    def predict(self, X):
        sig, rows = self._act(X)
        out = np.zeros(len(X), dtype=int)
        if rows is not None and len(rows):
            out[rows] = self.inner_.predict(X.iloc[rows])
        return out

    def predict_proba(self, X):
        sig, rows = self._act(X)
        p = np.zeros(len(X), dtype=float)
        if rows is not None and len(rows):
            p[rows] = self.inner_.predict_proba(X.iloc[rows])[:, 1]
        return np.column_stack([1.0 - p, p])


def model_zoo(seed):
    """The five models of project_details.md, simplest first.

    No hyperparameter search. The effective sample is roughly 700 observations
    per ticker, a grid would multiply the trial counter the Deflated Sharpe
    Ratio depends on, and the 2026-11-10 scope decision says one number per
    question. The settings below are fixed, deliberately conservative, and
    chosen for the sample size rather than tuned on it -- which is why they are
    a stated choice rather than a search result.
    """
    return {
        "buy_and_hold": AlwaysAct(),
        "ma_crossover": MACrossover(),
        "logistic": make_pipeline(
            LogisticRegression(max_iter=1000, random_state=seed)
        ),
        "gbt": make_pipeline(HistGradientBoostingClassifier(
            # Capacity held down on purpose: ~700 effective observations is the
            # binding constraint, not model class. Shallow trees, a big leaf
            # floor and explicit L2 are all the same decision.
            max_leaf_nodes=7,
            min_samples_leaf=50,
            learning_rate=0.05,
            max_iter=200,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
        )),
        "meta_crossover_logistic": MetaLabel(
            make_pipeline(LogisticRegression(max_iter=1000, random_state=seed))
        ),
    }
