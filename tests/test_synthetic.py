import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.cv import PurgedKFold
from src.evaluate import cross_val_scores, summarise
from src.features import design_matrix, make_pipeline
from src.purging import embargo_bars
from src.synthetic import synthetic_dataset, synthetic_ohlc

RETURNS = ["ret_1d", "ret_5d", "ret_10d", "ret_20d"]
N_BARS = 1400
SEEDS = (1, 2, 3)


def auc(phi, seed, columns=None):
    df, bars = synthetic_dataset(n_bars=N_BARS, phi=phi, seed=seed)
    X, y, t1 = design_matrix(df)
    if columns:
        X = X[columns]
    cv = PurgedKFold(t1, bars, 5, embargo_bars(len(t1), 0.01))
    pipe = make_pipeline(LogisticRegression(max_iter=1000, random_state=0))
    return summarise(cross_val_scores(pipe, X, y, cv))["auc"]


def mean_auc(phi, columns=None):
    return float(np.mean([auc(phi, s, columns) for s in SEEDS]))


def test_adjustment_factor_is_exactly_one():
    # Adj Close == Close by construction, so the high/low scaling path is a
    # no-op and cannot confound the check.
    raw = synthetic_ohlc(n_bars=300, phi=0.0, seed=0)
    assert (raw["Adj Close"] == raw["Close"]).all()


def test_ar1_coefficient_actually_lands_in_the_series():
    # If the generator did not plant what it claims, every result below is void.
    flat = synthetic_ohlc(n_bars=6000, phi=0.0, seed=0)["Close"]
    stiff = synthetic_ohlc(n_bars=6000, phi=0.6, seed=0)["Close"]

    r_flat = np.diff(np.log(flat.to_numpy()))
    r_stiff = np.diff(np.log(stiff.to_numpy()))

    assert abs(np.corrcoef(r_flat[1:], r_flat[:-1])[0, 1]) < 0.05
    assert np.corrcoef(r_stiff[1:], r_stiff[:-1])[0, 1] > 0.45


def test_synthetic_resembles_the_real_labelled_data():
    # If the synthetic label distribution were wildly different, the check
    # would be validating a pipeline we do not actually run.
    df, bars = synthetic_dataset(n_bars=3000, phi=0.0, seed=1)
    pos = {d: i for i, d in enumerate(bars)}
    window = (df["t1"].map(pos) - df.index.map(pos)).mean()

    assert 0.30 < df["label"].mean() < 0.55      # real data: ~0.40
    assert 3.0 < window < 9.0                    # real data: ~6 bars


def test_no_direction_signal_is_invented_on_a_random_walk():
    # The failure that would void every later result: beating chance on noise.
    assert abs(mean_auc(0.0, RETURNS) - 0.50) < 0.05


def test_planted_signal_is_recovered():
    # phi=0.8 rather than 0.6, so three seeds are enough to be stable.
    assert mean_auc(0.8, RETURNS) - mean_auc(0.0, RETURNS) > 0.10


def test_label_leaks_through_volatility_even_with_no_price_signal():
    # The floor is not 0.50. sigma_t sets the barrier width, wide barriers are
    # less likely to be touched, an untouched barrier times out, and a timeout
    # is label 0 -- so trailing volatility predicts the label on a random walk.
    df, bars = synthetic_dataset(n_bars=3000, phi=0.0, seed=1)
    pos = {d: i for i, d in enumerate(bars)}
    timeout = (df["t1"].map(pos) - df.index.map(pos)) == 10

    assert np.corrcoef(df["realized_vol_20d"], timeout.astype(float))[0, 1] > 0.10
    assert mean_auc(0.0) > mean_auc(0.0, RETURNS)


def test_direction_is_a_coin_flip_among_rows_that_touched_a_barrier():
    # The cleanest statement that the random walk has no directional edge: of
    # the rows that resolved by touching something, half went up.
    df, bars = synthetic_dataset(n_bars=3000, phi=0.0, seed=1)
    pos = {d: i for i, d in enumerate(bars)}
    touched = (df["t1"].map(pos) - df.index.map(pos)) != 10

    assert abs(df.loc[touched, "label"].mean() - 0.50) < 0.04
