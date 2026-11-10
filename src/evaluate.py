"""Scoring one estimator under one splitter. Shared by step 10 and step 6.

Deliberately thin. It fits the pipeline inside each fold -- which is the only
place it may be fitted -- and returns per-fold numbers rather than a single
mean, because on an effective sample of a few hundred the spread between folds
is part of the result rather than noise to be averaged away.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, roc_auc_score


def cross_val_scores(pipeline, X, y, cv):
    """Per-fold AUC, precision and context, fitting inside every fold.

    AUC is the headline here because it is threshold-free and its chance level
    is exactly 0.5, which is what the synthetic check needs to compare against.
    Precision is reported beside the majority-class rate, per
    project_details.md: a precision of 0.60 means nothing until you know
    whether always-act would have scored 0.60 too.

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

        rows.append({
            "fold": fold,
            "n_train": len(train),
            "n_test": len(test),
            "auc": roc_auc_score(truth, proba) if truth.nunique() > 1 else np.nan,
            "precision": precision_score(truth, pred, zero_division=0.0),
            "majority_rate": max(truth.mean(), 1 - truth.mean()),
            "positive_rate": float(truth.mean()),
        })

    return pd.DataFrame(rows)


def summarise(scores):
    """Mean AUC with its standard error across folds, and mean precision."""
    auc = scores["auc"].dropna()
    return {
        "auc": float(auc.mean()),
        "auc_se": float(auc.std(ddof=1) / np.sqrt(len(auc))) if len(auc) > 1 else np.nan,
        "precision": float(scores["precision"].mean()),
        "majority_rate": float(scores["majority_rate"].mean()),
        "n_train": int(scores["n_train"].mean()),
    }
