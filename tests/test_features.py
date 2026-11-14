import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.cv import PurgedKFold
from src.features import (EXCLUDED, FEATURES, design_matrix, make_pipeline,
                          pre_holdout)


def frame(n=200, horizon=10, periods=260, seed=0):
    """A synthetic labelled frame with the same columns as the real files."""
    r = np.random.default_rng(seed)
    bars = pd.bdate_range("2026-01-05", periods=periods)
    df = pd.DataFrame({c: r.normal(0, 1, n) for c in FEATURES}, index=bars[:n])
    for c in EXCLUDED:
        df[c] = r.normal(100, 1, n)
    df["label"] = r.integers(0, 2, n)
    df["t1"] = [bars[i + horizon] for i in range(n)]
    return df, bars


def test_design_matrix_holds_exactly_the_feature_list():
    df, _ = frame()
    X, y, t1 = design_matrix(df)

    assert list(X.columns) == FEATURES
    assert not any(c in X.columns for c in EXCLUDED)
    assert y.dtype == int
    assert len(X) == len(y) == len(t1)


def test_design_matrix_rejects_a_missing_feature():
    df, _ = frame()
    with pytest.raises(ValueError, match="missing features"):
        design_matrix(df.drop(columns=["vol_zscore_20d"]))


def test_unshifted_columns_stay_out():
    # The ones most likely to be let in by accident, since they look harmless.
    for col in ("log_return", "sigma", "Adj Close", "upper_barrier"):
        assert col in EXCLUDED
        assert col not in FEATURES


def test_scaler_is_fitted_on_training_rows_only():
    # The property the step exists for. If the scaler were fitted on everything
    # first, its mean would match the whole matrix rather than the fold's.
    df, bars = frame()
    X, y, t1 = design_matrix(df)
    pipe = make_pipeline(LogisticRegression())

    for train, _ in PurgedKFold(t1, bars, 5).split(X):
        pipe.fit(X.iloc[train], y.iloc[train])
        fitted = pipe.named_steps["scale"]

        assert np.allclose(fitted.mean_, X.iloc[train].mean().to_numpy())
        assert not np.allclose(fitted.mean_, X.mean().to_numpy())


def test_an_outlier_in_the_test_fold_cannot_move_the_scaler():
    # A canary. One absurd value, placed in rows that are only ever tested on.
    # If any test row reached `fit`, the scale would blow up and this fails.
    df, bars = frame()
    X, y, t1 = design_matrix(df)
    cv = PurgedKFold(t1, bars, 5)

    train, test = next(iter(cv.split(X)))
    clean = make_pipeline(LogisticRegression())
    clean.fit(X.iloc[train], y.iloc[train])

    spiked = X.copy()
    spiked.iloc[test, 0] = 1e9
    dirty = make_pipeline(LogisticRegression())
    dirty.fit(spiked.iloc[train], y.iloc[train])

    assert np.allclose(clean.named_steps["scale"].scale_,
                       dirty.named_steps["scale"].scale_)


def test_pipeline_order_is_impute_scale_model():
    pipe = make_pipeline(LogisticRegression())
    assert list(pipe.named_steps) == ["impute", "scale", "model"]


def test_imputer_is_a_no_op_on_the_current_feature_set():
    # Documented as insurance rather than active. If a future feature or a
    # wider lookback reintroduces NaNs, this fails and the claim gets revisited
    # instead of silently becoming false.
    df, _ = frame()
    X, _, _ = design_matrix(df)
    assert not X.isna().any().any()


def test_pre_holdout_purges_rows_resolving_into_the_holdout():
    # A date filter would keep the last few rows of 2022, whose labels were
    # decided by 2023 bars.
    df, bars = frame(n=200, horizon=10, periods=260)
    holdout_start = df.index[150]

    kept = pre_holdout(df, holdout_start, bars)
    naive = df.loc[df.index < holdout_start]

    assert len(kept) < len(naive)
    assert (kept["t1"] < holdout_start).all()
    assert kept.index.max() < holdout_start


def test_pre_holdout_keeps_everything_when_nothing_crosses():
    df, bars = frame(n=200, horizon=10, periods=260)
    # A boundary far past every t1 leaves the whole frame intact.
    kept = pre_holdout(df, bars[259], bars)
    assert len(kept) == len(df)
