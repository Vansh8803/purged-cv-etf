"""Scoring one estimator under one splitter.

Thin on purpose. It fits the pipeline inside each fold, which is the only place
it may be fitted, and returns per-fold numbers rather than one mean: on a few
hundred effective observations the spread between folds is part of the result.

Precision is pooled across folds, not averaged. Our models act on as little as
11% of bars and the per-fold action count swings tenfold inside one ticker, so
averaging ratios lets a 22-action fold outvote a 225-action one.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, roc_auc_score


def cross_val_scores(pipeline, X, y, cv):
    """Per-fold AUC and precision counts, fitting inside every fold.

    Precision comes back as raw counts (`n_acts`, `n_true_pos`) so `summarise`
    can pool them, with `coverage` beside it since precision at 11% coverage is
    a different claim from precision at 100%.

    `positive_rate` is what precision is read against. It is what always-act
    scores, and it is *not* `majority_rate`: at 40% positives those are 0.40
    and 0.60.

    `pipeline` is cloned per fold, so one call cannot leave a fitted estimator
    behind for the next.
    """
    from sklearn.base import clone

    rows = []
    for fold, (train, test) in enumerate(cv.split(X)):
        est = clone(pipeline)
        est.fit(X.iloc[train], y.iloc[train])

        proba = est.predict_proba(X.iloc[test])[:, 1]
        pred = est.predict(X.iloc[test])
        truth = y.iloc[test]
        acted = pred == 1

        rows.append({
            "fold": fold,
            "n_train": len(train),
            "n_test": len(test),
            "auc": roc_auc_score(truth, proba) if truth.nunique() > 1 else np.nan,
            "precision_fold": precision_score(truth, pred, zero_division=0.0),
            "n_acts": int(acted.sum()),
            "n_true_pos": int((acted & (truth.to_numpy() == 1)).sum()),
            "n_pos": int((truth == 1).sum()),
            "majority_rate": max(truth.mean(), 1 - truth.mean()),
            "positive_rate": float(truth.mean()),
        })

    return pd.DataFrame(rows)


def summarise(scores):
    """Mean AUC with its standard error, and precision pooled over folds."""
    auc = scores["auc"].dropna()
    acts = int(scores["n_acts"].sum())
    tested = int(scores["n_test"].sum())

    return {
        "auc": float(auc.mean()),
        "auc_se": float(auc.std(ddof=1) / np.sqrt(len(auc))) if len(auc) > 1 else np.nan,
        # Pooled: total true positives over total actions.
        "precision": float(scores["n_true_pos"].sum() / acts) if acts else 0.0,
        "precision_fold_mean": float(scores["precision_fold"].mean()),
        "coverage": acts / tested if tested else 0.0,
        # What precision is read against, weighted by fold size.
        "positive_rate": float(scores["n_pos"].sum() / tested) if tested else np.nan,
        "majority_rate": float(scores["majority_rate"].mean()),
        "n_train": int(scores["n_train"].mean()),
    }


def paired_gap(folds, left, right, value="auc"):
    """Mean difference between two schemes, with its *paired* standard error.

    Both schemes split the same rows the same way, so their test folds are
    identical and only the training sets differ. That makes the comparison
    paired: the noise on it is the spread of the per-fold differences, about
    ten times tighter than either scheme's own error.

    `folds` is the per-fold frame with `ticker`, `model`, `fold` and `scheme`.
    Returns a frame indexed by model with the gap, its paired error, and t.
    """
    wide = folds.pivot_table(index=["ticker", "model", "fold"],
                             columns="scheme", values=value)
    if left not in wide or right not in wide:
        raise ValueError(f"need both {left!r} and {right!r} in scheme column")

    diff = (wide[left] - wide[right]).dropna()
    grouped = diff.groupby(level="model")

    out = pd.DataFrame({
        "gap": grouped.mean(),
        "paired_se": grouped.std(ddof=1) / np.sqrt(grouped.count()),
        "n": grouped.count(),
    })
    # A rule scores identically under every scheme, so its gap is zero with
    # zero spread. That is a degenerate t, not a significant one.
    out["t"] = np.where(out["paired_se"] > 0, out["gap"] / out["paired_se"], np.nan)
    return out


def oof_predictions(pipeline, X, y, cv):
    """Out-of-fold predictions for every row, as the backtest needs them.

    Each row is predicted by a model that never trained on it. Four of the five
    folds still train partly on the future, so what this produces is a
    cross-validated backtest rather than a simulation of trading. Walk-forward
    is the scheme that does not do this.

    Returns `pred`, `proba` and the fold each row came from, indexed as `X`.
    """
    from sklearn.base import clone

    pred = pd.Series(np.nan, index=X.index, dtype=float)
    proba = pd.Series(np.nan, index=X.index, dtype=float)
    fold_of = pd.Series(np.nan, index=X.index, dtype=float)

    for fold, (train, test) in enumerate(cv.split(X)):
        est = clone(pipeline).fit(X.iloc[train], y.iloc[train])
        pred.iloc[test] = est.predict(X.iloc[test])
        proba.iloc[test] = est.predict_proba(X.iloc[test])[:, 1]
        fold_of.iloc[test] = fold

    return pd.DataFrame({"pred": pred, "proba": proba, "fold": fold_of})
