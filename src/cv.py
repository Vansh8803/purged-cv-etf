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


class WalkForward:
    """Expanding-window walk-forward. Step 4: the corroborating scheme.

    The sample is cut into `n_splits + 1` contiguous blocks in time order. Fold
    *i* trains on every block up to and including *i* and tests on block
    *i + 1*, so the training set grows with each fold and never contains a bar
    later than the block being tested. The first block is only ever trained on,
    which is why there is one more block than there are folds.

    This is the corroboration, not the primary number. PurgedKFold trains on
    future data for four of its five folds -- legitimate for measuring
    generalisation, but not something a live strategy could do. If the purged
    5-fold result and this one agree, the result is not an artifact of one
    split's geometry. If they disagree, the ordering is doing the work and that
    is itself worth reporting.

    Purging still applies. Training on the past does not by itself prevent a
    leak: the last training rows before a test block have label windows running
    up to `horizon` bars forward, straight into it. That is the same overlap
    PurgedKFold removes, so it goes through the same `purge_mask`.

    **The embargo is a no-op here, deliberately.** It drops rows starting in
    `(test_end, test_end + embargo]`, and an expanding window has no training
    row starting after its test block -- that is the whole point of the scheme.
    The parameter is accepted so the two splitters are interchangeable in the
    sweep, and the test suite pins the no-op so it reads as a property rather
    than an omission. It means walk-forward is flat across the embargo sweep,
    and the write-up should say why rather than leave a flat line unexplained.

    Parameters are the same as PurgedKFold's.
    """

    def __init__(self, t1, bar_index, n_splits=5, embargo=0):
        if n_splits < 1:
            raise ValueError("n_splits must be at least 1")
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
        """Yield (train_positions, test_positions), training set expanding."""
        n = len(self.t1)
        if X is not None and len(X) != n:
            raise ValueError(f"X has {len(X)} rows, t1 has {n}")
        if self.n_splits + 1 > n:
            raise ValueError(f"n_splits={self.n_splits} needs {self.n_splits + 1} rows, has {n}")

        blocks = np.array_split(np.arange(n), self.n_splits + 1)

        for i in range(self.n_splits):
            test_pos = blocks[i + 1]
            start, end = block_span(self.t1.iloc[test_pos])
            keep = purge_mask(self.t1, start, end, self.bar_index, self.embargo)

            # Everything strictly before this test block, then purged.
            past = np.zeros(n, dtype=bool)
            past[np.concatenate(blocks[:i + 1])] = True

            yield np.flatnonzero(past & keep.to_numpy()), test_pos


class NaiveKFold:
    """The control. Standard K-fold, no purging, no embargo.

    Run deliberately and labelled a control everywhere it appears, per
    decision-log.md 2026-10-14. It is wrong for this data, and measuring what
    purging changes is impossible without it.

    `t1` and `bar_index` are accepted and then ignored, so the three schemes are
    interchangeable at step 6. Ignoring them *is* the control: it is exactly the
    information a careless pipeline throws away.

    `shuffle` is the whole fold-geometry question in one flag, and it is not a
    detail. Contiguous is the default because `sklearn.KFold` does not shuffle
    either, so it is what "standard 5-fold K-fold" means. Shuffling scatters
    each fold's rows across the whole sample, which puts a training row next to
    a test row almost everywhere and leaks far more -- 88% of training rows
    against 0.33%, measured on SPY. See decision-log.md, 2026-11-10.
    """

    def __init__(self, t1, bar_index=None, n_splits=5, shuffle=False, seed=None):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.t1 = t1
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.seed = seed

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X=None, y=None, groups=None):
        """Yield (train_positions, test_positions). Everything not in the test
        fold is trained on, which is the omission being measured."""
        n = len(self.t1)
        if X is not None and len(X) != n:
            raise ValueError(f"X has {len(X)} rows, t1 has {n}")
        if self.n_splits > n:
            raise ValueError(f"n_splits={self.n_splits} exceeds {n} rows")

        positions = np.arange(n)
        if self.shuffle:
            np.random.default_rng(self.seed).shuffle(positions)

        for fold in np.array_split(positions, self.n_splits):
            test_pos = np.sort(fold)
            train_pos = np.setdiff1d(np.arange(n), test_pos)
            yield train_pos, test_pos


def leakage(cv, t1):
    """How much of a scheme's training data overlaps the fold it is scored on.

    The number the headline rests on, so it lives here with a test rather than
    in a notebook. An earlier version of these figures came from a throwaway
    script and could not be reproduced, and a number that cannot be reproduced
    is worse in a write-up than in a pull request.

    Counted pairwise: each training row against each individual test label
    window, which is the definition. Not against the fold's [min t, max t1]
    interval -- that form is right for a contiguous block and meaningless for a
    shuffled one, where the interval spans nearly the whole sample and almost
    every training row counts as overlapping it whether or not it shares a
    window with any test label. That substitution is what produced the bad 99.9%
    figure.

    Returns (train_fraction, test_fraction), averaged over folds: the share of
    training rows overlapping some test label, and the share of test rows whose
    window is overlapped by some training row.
    """
    starts = pd.DatetimeIndex(t1.index).to_numpy()
    ends = pd.DatetimeIndex(t1.to_numpy()).to_numpy()
    train_fracs, test_fracs = [], []

    for train, test in cv.split():
        # (train x test) overlap matrix on label windows.
        hit = ((ends[train][:, None] >= starts[test][None, :])
               & (starts[train][:, None] <= ends[test][None, :]))
        train_fracs.append(hit.any(axis=1).mean())
        test_fracs.append(hit.any(axis=0).mean())

    return float(np.mean(train_fracs)), float(np.mean(test_fracs))
