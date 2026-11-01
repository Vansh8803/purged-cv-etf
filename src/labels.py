import pandas as pd
import numpy as np


def compute_barriers(df, barrier_mult=2.5, vol_lookback=20):
    """Attach sigma_t and the two absolute barrier price levels.

    sigma_t is the standard deviation of daily log returns over the previous
    vol_lookback days, up to and including t. Everything in this project is in
    log-return space, so the barriers are a log move off day t's close:

        upper = close_t * exp(+mult * sigma_t)
        lower = close_t * exp(-mult * sigma_t)

    That keeps the two barriers symmetric in the same units sigma was measured
    in. Using close * (1 +/- mult * sigma) would make the up-move and the
    down-move different sizes in log terms, which the label would then inherit.
    """
    df = df.copy()
    df['sigma'] = df['log_return'].rolling(window=vol_lookback).std()
    df['upper_barrier'] = df['Adj Close'] * np.exp(barrier_mult * df['sigma'])
    df['lower_barrier'] = df['Adj Close'] * np.exp(-barrier_mult * df['sigma'])
    return df


def compute_labels(df, horizon=10):
    # Needs 'adj_high', 'adj_low', 'upper_barrier', 'lower_barrier' already on df.
    # Barriers are absolute price levels set from day t's close.
    df = df.copy()
    n = len(df)

    # 1. Build the forward-looking matrices using vector shifts.
    # We grab the next `horizon` days of highs and lows all at once.
    forward_highs = np.column_stack([df['adj_high'].shift(-i) for i in range(1, horizon + 1)])
    forward_lows = np.column_stack([df['adj_low'].shift(-i) for i in range(1, horizon + 1)])

    # 2. Did the forward high/low cross the barrier set at day t?
    # Missing future bars are NaN, and NaN comparisons are False, so the tail
    # never registers a spurious touch.
    hit_upper = forward_highs >= df['upper_barrier'].values[:, None]
    hit_lower = forward_lows <= df['lower_barrier'].values[:, None]

    any_upper = hit_upper.any(axis=1)
    any_lower = hit_lower.any(axis=1)

    # 3. First column where each barrier is touched. argmax returns 0 when
    # nothing is True, so park the untouched rows on a sentinel past the window
    # rather than at column 0.
    upper_col = np.where(any_upper, hit_upper.argmax(axis=1), horizon)
    lower_col = np.where(any_lower, hit_lower.argmax(axis=1), horizon)

    # 4. Resolve the race. A tie means both barriers were touched on the same
    # bar; we take that as 1, because the upper barrier is the side we would
    # have acted on. See decision-log 2026-11-01.
    label = np.where(any_upper & (upper_col <= lower_col), 1, 0)

    # 5. t1, in TRADING bars off t. A touch resolves on its own bar; a timeout
    # resolves on t+horizon. Both are positions in the index, so t1 is always a
    # real trading date -- purging is defined in terms of it.
    touched = any_upper | any_lower
    first_col = np.minimum(upper_col, lower_col)
    offset = np.where(touched, first_col + 1, horizon)
    t1_pos = np.arange(n) + offset

    # 6. A row is only usable if its label actually resolved inside the data we
    # have: either a genuine touch, or a full horizon of bars to time out over.
    # The rows at the very end have neither and would otherwise be recorded as
    # timeouts purely because their future is missing.
    resolved = t1_pos <= n - 1

    df['label'] = label
    df['t1'] = pd.NaT
    df.loc[df.index[resolved], 't1'] = df.index[t1_pos[resolved]]

    return df.loc[resolved].copy()
