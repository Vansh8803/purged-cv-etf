"""Scoring one estimator under one splitter. Shared by step 10 and step 6.

Deliberately thin. It fits the pipeline inside each fold -- which is the only
place it may be fitted -- and returns per-fold numbers rather than a single
mean, because on an effective sample of a few hundred the spread between folds
is part of the result rather than noise to be averaged away.

**Precision is pooled, not averaged.** Our models act on as little as 11% of
bars, and the per-fold action count swings by an order of magnitude inside a
single ticker -- 157, 22, 170, 225, 50 for logistic on SPY. An unweighted mean
of per-fold precision gives the 22-action fold the same say as the 225-action
one, and since a small denominator is a noisy one that reliably reads high, the
mean comes out inflated. Worst for meta-labelling, whose entire claim is
precision. So the counts are summed first and divided once. See
decision-log.md, 2026-11-14.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, roc_auc_score


def cross_val_scores(pipeline, X, y, cv):
    """Per-fold AUC, precision counts and context, fitting inside every fold.

    AUC is the headline because it is threshold-free and its chance level is
    exactly 0.5, which is what the synthetic check compares against.

    Precision is returned as its raw counts -- `n_acts` and `n_true_pos` --
    alongside the per-fold ratio, so `summarise` can pool them. `coverage` is
    how often the model acted at all: precision at 11% coverage is a different
    claim from precision at 100%, and the pair has to travel together.

    `positive_rate` is the number precision is read against. It is what
    always-act scores by construction, and it is *not* `majority_rate` -- with
    40% positives the majority rate is 0.60 while the precision benchmark is
    0.40, so confusing them makes every model look like a failure.

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
        # Pooled: total true positives over total actions, not a mean of ratios.
        "precision": float(scores["n_true_pos"].sum() / acts) if acts else 0.0,
        "precision_fold_mean": float(scores["precision_fold"].mean()),
        "coverage": acts / tested if tested else 0.0,
        # The benchmark precision is read against, weighted by fold size.
        "positive_rate": float(scores["n_pos"].sum() / tested) if tested else np.nan,
        "majority_rate": float(scores["majority_rate"].mean()),
        "n_train": int(scores["n_train"].mean()),
    }


def paired_gap(folds, left, right, value="auc"):
    """Mean difference between two schemes, with its *paired* standard error.

    The control and the purged scheme are built from the same `np.array_split`
    of the same rows, so their test folds are identical -- only the training
    sets differ. That makes the comparison paired, and the noise on it is the
    spread of the per-fold *differences*, not the spread of either scheme's
    score. On our data the paired standard error is about ten times tighter
    than the unpaired one, so using the wrong denominator understates how
    precisely the gap has been measured.

    `folds` is the per-fold frame with `ticker`, `model`, `fold` and `scheme`
    columns. Returns a frame indexed by model, with the mean gap, its paired
    standard error, and t.
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
    # A rule scores identically under every scheme, so its gap is exactly zero
    # with zero spread. That is a degenerate t, not a significant one.
    out["t"] = np.where(out["paired_se"] > 0, out["gap"] / out["paired_se"], np.nan)
    return out
