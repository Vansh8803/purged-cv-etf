"""Cross-validation splitters. Step 3: PurgedKFold with an embargo.

Written rather than imported. It is about fifty lines, importing the one thing
the project is about would be absurd, and both of us have to be able to derive
it at a whiteboard -- see decision-log.md, 2026-10-14.
"""

import numpy as np
import pandas as pd

from src.purging import purge_mask, block_span


class PurgedKFold:
    """K-fold over time-ordered rows, with overlapping labels purged.

    Folds are contiguous blocks of rows in time order. Nothing is shuffled, but
    that on its own fixes nothing: the leak is in the label windows, so a
    training row immediately before a test fold still carries nine days of the
    fold's outcome whether or not the rows were shuffled. The purge is what
    removes it, and `src/purging.py` holds that rule.

    `t1` and `bar_index` are required and explicit, for the same reason
    `src/uniqueness.py` requires them: a window's length is only meaningful in
    bars, and the (t, t1) dates alone do not pin the calendar down.

    Duck-types scikit-learn's splitter protocol -- `split` and `get_n_splits` --
    so it drops into `cross_val_score` and `GridSearchCV` directly. It does not
    subclass `BaseCrossValidator`, which would buy nothing and tie us to
    sklearn's internals.

    Parameters
    ----------
    t1 : Series
        Indexed by the bar a label opens on, valued by the bar it resolves on.
        Must be sorted and free of NaT, which is what `compute_labels` returns.
    bar_index : DatetimeIndex
        The trading calendar the embargo is measured against.
    n_splits : int
        Number of folds.
    embargo : int
        Embargo length in *bars*, not days. `purging.embargo_bars` converts
        config's `embargo_frac`; the sensitivity sweep passes 0 to 20 directly.

    Usage, per ticker, since we fit one model per ETF:

        cv = PurgedKFold(t1, bars, n_splits=5, embargo=embargo_bars(len(t1), 0.01))
        for train_pos, test_pos in cv.split(X):
            ...
    """

    def __init__(self, t1, bar_index, n_splits=5, embargo=0):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if embargo < 0:
            raise ValueError("embargo must not be negative")
        if not t1.index.is_monotonic_increasing:
            raise ValueError("t1 must be sorted by its start date")
        if t1.index.has_duplicates:
            raise ValueError("t1 has duplicate start dates")

        self.t1 = t1
        self.bar_index = pd.DatetimeIndex(bar_index)
        self.n_splits = n_splits
        self.embargo = embargo

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X=None, y=None, groups=None):
        """Yield (train_positions, test_positions) as integer arrays.

        Positions index into `t1`, so whatever is passed as X must be in the
        same order and of the same length -- checked rather than assumed,
        because a silent misalignment here would corrupt every fold while
        looking perfectly healthy.
        """
        n = len(self.t1)
        if X is not None and len(X) != n:
            raise ValueError(f"X has {len(X)} rows, t1 has {n}")
        if self.n_splits > n:
            raise ValueError(f"n_splits={self.n_splits} exceeds {n} rows")

        positions = np.arange(n)

        for test_pos in np.array_split(positions, self.n_splits):
            start, end = block_span(self.t1.iloc[test_pos])
            keep = purge_mask(self.t1, start, end, self.bar_index, self.embargo)

            # Test rows are purged by construction: each has t <= end and
            # t1 >= start, so it overlaps its own fold. Nothing extra to do,
            # and the test suite should pin that rather than trust this note.
            yield positions[keep.to_numpy()], test_pos
