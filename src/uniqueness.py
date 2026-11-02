import pandas as pd
import numpy as np


def _windows(t1, bar_index):
    """Map each label's [t, t1] window onto integer positions on the bar axis.

    `bar_index` is the trading calendar the windows are measured against, and it
    is required rather than inferred. A window's length is only meaningful in
    bars, and the labels alone do not pin the calendar down: t1 can point past
    the last labelled row, and the set of (t, t1) dates can have gaps that would
    silently shorten every window measured against them.
    """
    ends = pd.DatetimeIndex(t1.values)
    bar_index = pd.DatetimeIndex(bar_index)

    start = bar_index.searchsorted(t1.index)
    end = bar_index.searchsorted(ends)
    if end.max() >= len(bar_index) or not (bar_index[end] == ends).all():
        raise ValueError("some t1 dates are not bars in bar_index")
    if not (bar_index[start] == t1.index).all():
        raise ValueError("some label start dates are not bars in bar_index")
    return bar_index, start, end


def concurrency(t1, bar_index):
    """How many label windows span each bar.

    A window is [t, t1] inclusive at both ends: the day the position would be
    opened and the day its outcome becomes known.
    """
    bar_index, start, end = _windows(t1, bar_index)

    # Difference array then cumulative sum, so this is O(bars) rather than
    # O(labels x bars).
    diff = np.zeros(len(bar_index) + 1)
    np.add.at(diff, start, 1)
    np.add.at(diff, end + 1, -1)

    return pd.Series(np.cumsum(diff)[:len(bar_index)], index=bar_index, name="concurrency")


def average_uniqueness(t1, bar_index):
    """Per-label uniqueness: the mean of 1 / concurrency across its own window.

    A label spanning days nobody else spans scores 1.0. A label every day of
    which it shares with exactly one other scores 0.5. This is the correction
    for overlapping labels not being independent observations.
    """
    bar_index, start, end = _windows(t1, bar_index)
    counts = concurrency(t1, bar_index).values

    inv = np.divide(1.0, counts, out=np.zeros_like(counts, dtype=float), where=counts > 0)
    cum = np.concatenate([[0.0], np.cumsum(inv)])

    # Mean of inv over [start, end] inclusive, every label at once.
    u = (cum[end + 1] - cum[start]) / (end - start + 1)
    return pd.Series(u, index=t1.index, name="uniqueness")


def effective_sample_size(t1, bar_index):
    """Average uniqueness times the row count.

    The honest denominator. 4,000 rows at 0.16 uniqueness carry roughly the
    information of 640 independent observations, and every standard error in
    the write-up should be read against this number rather than the raw one.
    """
    u = average_uniqueness(t1, bar_index)
    return float(u.mean()) * len(u)
