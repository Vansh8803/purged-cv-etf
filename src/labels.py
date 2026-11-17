import pandas as pd
import numpy as np


def compute_barriers(df, barrier_mult=2.5, vol_lookback=20):
    """Attach sigma_t and the two barrier price levels.

    sigma_t is the standard deviation of daily log returns over the previous
    `vol_lookback` days, up to and including t. We work in log-return space, so
    the barriers are a log move off the close:

        upper = close_t * exp(+mult * sigma_t)
        lower = close_t * exp(-mult * sigma_t)

    That keeps them symmetric in the units sigma was measured in. Using
    `close * (1 +/- mult * sigma)` would make the up and down moves different
    sizes in log terms, and the label would inherit the difference.
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

    # The next `horizon` days of highs and lows, all at once.
    forward_highs = np.column_stack([df['adj_high'].shift(-i) for i in range(1, horizon + 1)])
    forward_lows = np.column_stack([df['adj_low'].shift(-i) for i in range(1, horizon + 1)])

    # Did the forward high/low cross day t's barrier? Missing bars are NaN and
    # NaN comparisons are False, so the tail never registers a touch.
    hit_upper = forward_highs >= df['upper_barrier'].values[:, None]
    hit_lower = forward_lows <= df['lower_barrier'].values[:, None]

    any_upper = hit_upper.any(axis=1)
    any_lower = hit_lower.any(axis=1)

    # argmax returns 0 when nothing is True, so untouched rows go to a
    # sentinel past the window rather than to column 0.
    upper_col = np.where(any_upper, hit_upper.argmax(axis=1), horizon)
    lower_col = np.where(any_lower, hit_lower.argmax(axis=1), horizon)

    # Resolve the race. A tie means both barriers were touched on the same
    # bar, and we take that as 1 since the upper barrier is the side we would
    # have acted on.
    label = np.where(any_upper & (upper_col <= lower_col), 1, 0)

    # t1 in trading bars off t, as a position in the index, so it is always a
    # real bar. Purging is defined on it.
    touched = any_upper | any_lower
    first_col = np.minimum(upper_col, lower_col)
    offset = np.where(touched, first_col + 1, horizon)
    t1_pos = np.arange(n) + offset

    # A row is only usable if its label resolved inside the data we have. The
    # rows at the end have no future to look at and would otherwise be recorded
    # as timeouts.
    resolved = t1_pos <= n - 1

    df['label'] = label
    df['t1'] = pd.NaT
    df.loc[df.index[resolved], 't1'] = df.index[t1_pos[resolved]]

    return df.loc[resolved].copy()
