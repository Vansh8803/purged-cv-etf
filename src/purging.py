"""The overlap test the whole project turns on, written once.

Three different places hold data back and every one of them has to ask the same
question: given a block of bars we are about to test on, which rows are still
safe to train on? PurgedKFold (step 3), the walk-forward splitter (step 4) and
the holdout boundary (step 13) all call `purge_mask` rather than each
implementing the rule, because three slightly different versions of the one
idea the project is about is the worst outcome available to us.

Everything here is defined on label windows [t, t1], never on row order.
Shuffling is not what breaks ordinary K-fold on this data; the overlap in the
label windows is, and it leaks across a fold boundary whether or not the rows
were shuffled.
"""

import numpy as np
import pandas as pd


def _bar_position(bar_index, date, what):
    """Integer position of `date` on `bar_index`, or a clear error.

    Length in bars is only meaningful against the trading calendar, so a date
    that is not a bar is a bug in the caller rather than something to round.
    """
    bar_index = pd.DatetimeIndex(bar_index)
    pos = int(bar_index.searchsorted(pd.Timestamp(date)))
    if pos >= len(bar_index) or bar_index[pos] != pd.Timestamp(date):
        raise ValueError(f"{what} ({date}) is not a bar in bar_index")
    return pos


def embargo_bars(n_bars, frac):
    """Embargo length in bars, from `embargo_frac` in config.yaml.

    config states the embargo as a fraction of the sample rather than a bar
    count so that it scales with however much data a given run has. The
    sensitivity sweep overrides it with explicit bar counts from 0 to 20.
    """
    return int(round(n_bars * frac))


def block_span(test_t1):
    """The stretch of bars a test block actually occupies.

    Not [first test row, last test row]. A test label resolves up to `horizon`
    bars after its own row, so the block's reach is its last *resolution* date.
    Measuring the span by row dates alone leaves a gap: a training row starting
    between the final test row and that resolution date has a label window
    overlapping a test label, and would survive the purge while knowing part of
    the answer. Taking the maximum t1 closes it.

    `test_t1` is the t1 Series restricted to the test rows.
    """
    if len(test_t1) == 0:
        raise ValueError("test block is empty")
    return pd.Timestamp(test_t1.index[0]), pd.Timestamp(test_t1.max())


def purge_mask(t1, test_start, test_end, bar_index=None, embargo=0):
    """Rows safe to train on, given a test block spanning [test_start, test_end].

    Both rules come straight from project_details.md.

    **Purge.** Drop row i when its label window [t_i, t1_i] overlaps the test
    span at all, which is `t1_i >= test_start and t_i <= test_end`. A row whose
    outcome is still unresolved when the test period opens has been shown part
    of the outcome it is about to be scored against.

    **Embargo.** Additionally drop rows starting in
    `(test_end, test_end + embargo]` bars -- exclusive at the start, because
    test_end is already purged by the rule above, and inclusive at the end.
    Two reasons, and the second is the concrete one: prices are serially
    correlated, and the 20-day feature windows of a row just after the test
    block are computed from bars inside it. Such a row carries information
    about the test period even though its label window does not overlap it.

    `bar_index` is required when `embargo > 0`, since the embargo is a length in
    trading bars and the label dates alone do not pin the calendar down. It is
    passed explicitly for the same reason src/uniqueness.py demands it.

    Returns a boolean Series over `t1.index`: True means keep for training.

    The three callers differ only in what they pass:

    - PurgedKFold: `block_span()` of each fold, plus the embargo.
    - Walk-forward: `block_span()` of each forward block. Nothing trains on
      anything later, so the embargo only matters for the expanding window.
    - The holdout: `test_start` = holdout_start and `test_end` = the last bar,
      with no embargo, since nothing comes after the holdout. This is the purge
      that stops a December 2022 row whose label resolves in January 2023 from
      sitting in the training set.
    """
    if t1.isna().any():
        raise ValueError("t1 contains NaT; unresolved rows must be dropped first")

    starts = pd.DatetimeIndex(t1.index)
    ends = pd.DatetimeIndex(t1.to_numpy())
    test_start = pd.Timestamp(test_start)
    test_end = pd.Timestamp(test_end)
    if test_end < test_start:
        raise ValueError("test_end precedes test_start")

    # Interval overlap, both ends inclusive.
    keep = ~((ends >= test_start) & (starts <= test_end))

    if embargo > 0:
        if bar_index is None:
            raise ValueError("bar_index is required when embargo > 0")
        bar_index = pd.DatetimeIndex(bar_index)
        end_pos = _bar_position(bar_index, test_end, "test_end")
        # Clipped, so a test block at the very end of the sample embargoes
        # whatever bars remain rather than running off the calendar.
        last = min(end_pos + embargo, len(bar_index) - 1)
        embargoed = (starts > test_end) & (starts <= bar_index[last])
        keep &= ~embargoed

    return pd.Series(keep, index=t1.index, name="train")


def purged_holdout_mask(t1, holdout_start, bar_index):
    """Training rows that are safe given an untouched holdout from `holdout_start`.

    The same purge as everywhere else, with the test block being the holdout
    itself. Named separately only because step 13 reads better for it.
    """
    bar_index = pd.DatetimeIndex(bar_index)
    return purge_mask(t1, holdout_start, bar_index[-1], bar_index, embargo=0)
