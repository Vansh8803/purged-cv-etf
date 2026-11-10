from pathlib import Path

import pandas as pd
import numpy as np
import pytest
from src.labels import compute_barriers, compute_labels

ROOT = Path(__file__).resolve().parents[1]


def toy(periods=30):
    """Flat series at 100, barriers at 105 / 95, on a weekday calendar.

    Highs and lows sit at 101 / 99 so nothing touches unless a test moves it.
    """
    dates = pd.bdate_range(start='2026-10-01', periods=periods)
    df = pd.DataFrame(index=dates)
    df['adj_high'] = 101.0
    df['adj_low'] = 99.0
    df['upper_barrier'] = 105.0
    df['lower_barrier'] = 95.0
    return df, dates


def test_upper_lower_and_timeout():
    df, d = toy()
    df.loc[d[3], 'adj_high'] = 106.0    # upper touched 3 bars after d[0]
    df.loc[d[7], 'adj_low'] = 94.0      # lower touched 2 bars after d[5]
    out = compute_labels(df, horizon=10)

    assert out.loc[d[0], 'label'] == 1
    assert out.loc[d[0], 't1'] == d[3]

    assert out.loc[d[5], 'label'] == 0
    assert out.loc[d[5], 't1'] == d[7]

    # d[15] sees neither barrier, so it times out.
    assert out.loc[d[15], 'label'] == 0


def test_timeout_t1_is_ten_trading_bars():
    # The bug this guards: t + 10 CALENDAR days lands on a weekend and is not a
    # date in the index at all. Purging is defined on t1, so it has to be a bar.
    df, d = toy()
    out = compute_labels(df, horizon=10)

    assert out.loc[d[0], 't1'] == d[10]
    # Every t1 is a real bar. It may sit past the last labelled row -- a row
    # near the end can resolve on a bar that is itself too late to label.
    assert out['t1'].isin(d).all()


def test_same_bar_tie_is_labelled_one():
    # Both barriers touched on the same bar: we call it 1, because the upper
    # barrier is the side we would have acted on.
    df, d = toy()
    df.loc[d[3], 'adj_high'] = 106.0
    df.loc[d[3], 'adj_low'] = 94.0
    out = compute_labels(df, horizon=10)

    assert out.loc[d[0], 'label'] == 1
    assert out.loc[d[0], 't1'] == d[3]


def test_touch_on_first_bar():
    df, d = toy()
    df.loc[d[1], 'adj_high'] = 106.0
    out = compute_labels(df, horizon=10)

    assert out.loc[d[0], 'label'] == 1
    assert out.loc[d[0], 't1'] == d[1]


def test_touch_beyond_horizon_is_a_timeout():
    # 11 bars out is outside a 10-bar window and must not count.
    df, d = toy()
    df.loc[d[11], 'adj_high'] = 106.0
    out = compute_labels(df, horizon=10)

    assert out.loc[d[0], 'label'] == 0
    assert out.loc[d[0], 't1'] == d[10]


def test_unresolved_tail_rows_are_dropped():
    # The last rows have no forward bars, so they cannot time out honestly.
    # A row that genuinely touches inside the data we have is still kept.
    df, d = toy(periods=15)
    df.loc[d[14], 'adj_high'] = 106.0   # d[13] touches one bar later
    out = compute_labels(df, horizon=10)

    assert out.index[-1] == d[13]
    assert out.loc[d[13], 'label'] == 1
    assert out.loc[d[13], 't1'] == d[14]
    assert d[14] not in out.index


def test_barriers_are_symmetric_log_moves():
    df = pd.DataFrame({
        'Adj Close': [100.0] * 25,
        'log_return': [0.01, -0.01] * 12 + [0.01],
    }, index=pd.bdate_range('2026-10-01', periods=25))
    out = compute_barriers(df, barrier_mult=2.5, vol_lookback=20)

    row = out.dropna(subset=['sigma']).iloc[0]
    assert np.isclose(row['upper_barrier'] * row['lower_barrier'], 100.0 ** 2)
    assert row['lower_barrier'] < 100.0 < row['upper_barrier']


def test_gld_adjustment_factor_is_one():
    # GLD distributes nothing, so Adj Close == Close and the scaled high/low
    # must equal the raw high/low exactly. A bug in the scaling path shows up
    # here as a difference between GLD and the other four.
    raw = pd.read_csv(
        ROOT / 'data/raw/GLD_2005-01-01_2025-12-31_dl2026-10-16.csv',
        index_col='Date', parse_dates=True,
    )
    factor = raw['Adj Close'] / raw['Close']
    assert (factor == 1.0).all()

    # Same scaling data_process.py applies. Derived from raw here on purpose:
    # data/processed/ is gitignored, so the test must not depend on it.
    assert np.allclose(raw['High'] * factor, raw['High'])
    assert np.allclose(raw['Low'] * factor, raw['Low'])


def brute_force(df, horizon=10):
    """The obviously-correct slow version, for the vectorised one to match.

    Plain loops, no numpy tricks, verifiable by reading it. `compute_labels`
    builds a (rows x horizon) matrix and resolves every row at once, which is
    fast but easy to get subtly wrong -- `argmax` returns 0 on an all-False
    row, and column k means bar t+k+1. Diffing against this converts "is the
    vectorised logic right" into "do the two agree", which a test can answer.
    """
    hi, lo = df['adj_high'].values, df['adj_low'].values
    ub, lb = df['upper_barrier'].values, df['lower_barrier'].values
    n = len(df)
    rows = []

    for t in range(n):
        label, t1 = None, None
        for k in range(1, horizon + 1):
            if t + k >= n:
                break                                   # ran out of bars
            up, dn = hi[t + k] >= ub[t], lo[t + k] <= lb[t]
            if up:
                label, t1 = 1, t + k; break             # covers the tie: 1
            if dn:
                label, t1 = 0, t + k; break
        if label is None:
            if t + horizon > n - 1:
                continue                                # unresolvable, dropped
            label, t1 = 0, t + horizon                  # genuine timeout
        rows.append((df.index[t], label, df.index[t1]))

    return pd.DataFrame(rows, columns=['Date', 'label', 't1']).set_index('Date')


def random_bars(seed, n, tight=True):
    """A random walk with barriers tight enough to force frequent ties."""
    r = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(r.normal(0, 0.02, n)))
    width = np.abs(r.normal(0, 0.03, n))
    mult = 1.005 if tight else 1.05
    return pd.DataFrame({
        'adj_high': close * (1 + width),
        'adj_low': close * (1 - width),
        'upper_barrier': close * mult,
        'lower_barrier': close / mult,
    }, index=pd.bdate_range('2026-01-01', periods=n))


@pytest.mark.parametrize('seed', range(25))
def test_matches_brute_force_on_random_series(seed):
    # Tight barriers on purpose: at the real 2.5-sigma width a same-bar tie is
    # rare, so random series with wide barriers would rarely exercise it.
    df = random_bars(seed, n=40)
    fast, slow = compute_labels(df, horizon=10), brute_force(df, horizon=10)

    assert fast.index.equals(slow.index)
    assert (fast['label'].values == slow['label'].values).all()
    assert (pd.to_datetime(fast['t1']).values == slow['t1'].values).all()


@pytest.mark.parametrize('n,horizon', [(11, 10), (10, 10), (9, 10), (25, 3), (4, 10)])
def test_matches_brute_force_when_n_differs_from_horizon(n, horizon):
    # A square (rows x horizon) matrix can hide a transposed broadcast, so the
    # shapes are deliberately unequal in both directions.
    df = random_bars(n * 100 + horizon, n=n)
    fast, slow = compute_labels(df, horizon=horizon), brute_force(df, horizon=horizon)

    assert fast.index.equals(slow.index)
    assert (fast['label'].values == slow['label'].values).all()
    assert (pd.to_datetime(fast['t1']).values == slow['t1'].values).all()
