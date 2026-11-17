"""The five models, and the rules they have to beat.

All five predict the same binary target, so precision and AUC mean the same
thing in every row of the comparison.

Two of them are not fitted at all. Dressing a rule as a classifier lets it go
through the same scoring path as a learned model -- same folds, same test rows,
same metric, same purging -- which is the only way the comparison is like for
like.

The rules read the raw design matrix rather than the scaled pipeline.
Standardising gives each column its own mean and standard deviation, which does
not preserve a comparison *between* two columns, and the crossover is exactly
that comparison.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src.features import make_pipeline


def crossover_signal(X):
    """True where the 20-day moving average sits above the 50-day.

    Read off the distance-from-MA features rather than recomputed from price,
    so the rule inherits the same one-bar shift as everything else and the lag
    is structural rather than something to remember.

    Watch the direction. With `d20 = (C - MA20) / MA20` we have
    `MA20 = C / (1 + d20)`, and since price and every average are positive:

        MA20 > MA50  <=>  C / (1 + d20) > C / (1 + d50)  <=>  d20 < d50.

    The inequality flips. Written the intuitive way round this would be a death
    cross and would still produce a plausible score.
    """
    return (X["dist_ma_20d"] < X["dist_ma_50d"]).to_numpy()


class _Rule(BaseEstimator, ClassifierMixin):
    """A deterministic rule wearing the estimator interface.

    `fit` ignores its training data on purpose, which is what makes these
    reference points: a rule cannot leak or overfit and scores identically
    under every scheme, so any control-versus-purged difference a learned model
    shows is the learning rather than the data.
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
    """Buy and hold as a classifier: act on every bar.

    Its precision is the positive rate by construction, which is the number
    every other precision is read against. Its AUC is 0.5, since it never
    discriminates.
    """

    def _signal(self, X):
        return np.ones(len(X), dtype=bool)


class MACrossover(_Rule):
    """The baseline: act while the 20-day MA is above the 50-day, lagged a bar."""

    def _signal(self, X):
        return crossover_signal(X)


class MetaLabel(BaseEstimator, ClassifierMixin):
    """The crossover picks direction; the model decides whether to act.

    Which way price will move is close to hopeless on daily bars. Whether to
    take a signal somebody else generated is a narrower question on a smaller,
    more homogeneous sample.

    Trained only on rows where the rule fires and predicts 0 everywhere else,
    so it can never act where the rule would not have.

    A fold where the rule never fires, or fires on one class only, leaves
    nothing to learn. Rather than crash or fit something degenerate it falls
    back to predicting 0 and records that in `fitted_`.
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
    """The five models, simplest first.

    No hyperparameter search. The effective sample is roughly 660 observations
    per ticker, and a grid would multiply the trial count the Deflated Sharpe
    Ratio depends on. These settings are fixed and chosen for the sample size
    rather than tuned on it, so they are a stated choice rather than a search
    result.
    """
    return {
        "buy_and_hold": AlwaysAct(),
        "ma_crossover": MACrossover(),
        "logistic": make_pipeline(
            LogisticRegression(max_iter=1000, random_state=seed)
        ),
        "gbt": make_pipeline(HistGradientBoostingClassifier(
            # Capacity held down on purpose: ~660 effective observations is
            # the binding constraint, not model class.
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
