"""The overlap rule, written once.

PurgedKFold, the walk-forward splitter and the holdout boundary all need the
same answer: given a block of bars we are about to test on, which rows are
still safe to train on? They all call `purge_mask` so the three cannot drift
apart.

Everything here works on label windows [t, t1], never on row order.
"""

import pandas as pd


def _bar_position(bar_index, date, what):
    """Integer position of `date` on `bar_index`. Raises if it is not a bar."""
    bar_index = pd.DatetimeIndex(bar_index)
    pos = int(bar_index.searchsorted(pd.Timestamp(date)))
    if pos >= len(bar_index) or bar_index[pos] != pd.Timestamp(date):
        raise ValueError(f"{what} ({date}) is not a bar in bar_index")
    return pos


def embargo_bars(n_bars, frac):
    """Embargo length in bars from `embargo_frac`, so it scales with the sample."""
    return int(round(n_bars * frac))


def block_span(test_t1):
    """The stretch of bars a test block occupies: (first t, largest t1).

    Not the last row's date. A test label resolves up to `horizon` bars after
    its own row, so the block reaches past its final row, and a training row
    starting in that tail would survive a purge measured on row dates alone.

    `test_t1` need not be sorted. t1 is not monotonic, so the span comes from
    the minimum start and the maximum t1 rather than the first and last
    elements.
    """
    if len(test_t1) == 0:
        raise ValueError("test block is empty")
    return pd.Timestamp(test_t1.index.min()), pd.Timestamp(test_t1.max())


def purge_mask(t1, test_start, test_end, bar_index=None, embargo=0):
    """Rows safe to train on, given a test block over [test_start, test_end].

    Purge: drop row i when its window [t_i, t1_i] overlaps the test span,
    which is `t1_i >= test_start and t_i <= test_end`.

    Embargo: also drop rows starting in `(test_end, test_end + embargo]`,
    exclusive at the start because test_end is already purged above. Our
    features use 20-day rolling windows, so a row just after the block
    computes its features partly from bars inside it.

    `bar_index` is required when `embargo > 0`, since the embargo is a length
    in trading bars and the label dates alone do not pin down the calendar.

    Returns a boolean Series over `t1.index`; True means keep for training.
    """
    if t1.isna().any():
        raise ValueError("t1 contains NaT; unresolved rows must be dropped first")

    starts = pd.DatetimeIndex(t1.index)
    ends = pd.DatetimeIndex(t1.to_numpy())
    test_start = pd.Timestamp(test_start)
    test_end = pd.Timestamp(test_end)
    if test_end < test_start:
        raise ValueError("test_end precedes test_start")

    keep = ~((ends >= test_start) & (starts <= test_end))

    if embargo > 0:
        if bar_index is None:
            raise ValueError("bar_index is required when embargo > 0")
        bar_index = pd.DatetimeIndex(bar_index)
        end_pos = _bar_position(bar_index, test_end, "test_end")
        # Clipped, so a block at the end of the sample embargoes what is left
        # rather than running off the calendar.
        last = min(end_pos + embargo, len(bar_index) - 1)
        keep &= ~((starts > test_end) & (starts <= bar_index[last]))

    return pd.Series(keep, index=t1.index, name="train")


def purged_holdout_mask(t1, holdout_start, bar_index):
    """Training rows that are safe given an untouched holdout.

    The same purge with the holdout as the test block, which is what stops a
    December row whose label resolves in January from reaching training.
    """
    bar_index = pd.DatetimeIndex(bar_index)
    return purge_mask(t1, holdout_start, bar_index[-1], bar_index, embargo=0)
