"""The three cross-validation schemes, all built on `src/purging.py`.

We wrote these rather than importing mlfinlab, since purging is the subject of
the project.
"""

import numpy as np
import pandas as pd

from src.purging import purge_mask, block_span


class PurgedKFold:
    """K-fold over time-ordered rows, with overlapping labels purged.

    Folds are contiguous blocks of rows. Not shuffling is not enough on its
    own: the leak is in the label windows, so a training row just before a
    fold still carries most of that fold's outcome either way.

    `t1` is indexed by the bar a label opens on and valued by the bar it
    resolves on, sorted and free of NaT. `bar_index` is the trading calendar
    the embargo is measured against; it is explicit because a window's length
    only means something in bars. `embargo` is in bars, not days.

    Duck-types sklearn's splitter protocol, so it drops into `cross_val_score`
    directly.

        cv = PurgedKFold(t1, bars, 5, embargo_bars(len(t1), 0.01))
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

        Positions index into `t1`, so X must be in the same order and length.
        We check rather than assume: a silent misalignment would corrupt every
        fold while looking healthy.
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

            # Test rows purge themselves: each has t <= end and t1 >= start,
            # so it overlaps its own fold.
            yield positions[keep.to_numpy()], test_pos


class WalkForward:
    """Expanding-window walk-forward, our corroborating scheme.

    The sample is cut into `n_splits + 1` blocks. Fold *i* trains on blocks
    0..*i* and tests on block *i+1*, so training grows each fold and never
    contains a bar later than the block being tested. The first block is only
    ever trained on, hence one more block than folds.

    Purging still applies: the last training rows before a block resolve into
    it, which is the same overlap PurgedKFold removes.

    The embargo is a no-op here, since an expanding window has no training row
    starting after its test block. We keep the parameter so the splitters stay
    interchangeable, and a test pins the no-op. It does mean walk-forward is
    flat across the embargo sweep.

    Parameters are PurgedKFold's.
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

            # Everything before this block, then purged.
            past = np.zeros(n, dtype=bool)
            past[np.concatenate(blocks[:i + 1])] = True

            yield np.flatnonzero(past & keep.to_numpy()), test_pos


class NaiveKFold:
    """The control. Standard K-fold, no purging, no embargo.

    Wrong for this data, which is why we run it: without it we cannot measure
    what purging changes. We label it a control everywhere it appears.

    `t1` and `bar_index` are accepted and ignored. Ignoring them *is* the
    control, and it keeps the three schemes interchangeable.

    `shuffle` is the fold-geometry question in one flag. Contiguous is the
    default because `sklearn.KFold` does not shuffle either. Shuffling
    scatters each fold across the whole sample and leaks 88% of training rows
    against 0.33%.
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
        """Yield (train_positions, test_positions). Everything outside the
        test fold is trained on, which is the omission we are measuring."""
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

    Our headline rests on this, so it lives in the module with a test rather
    than in a notebook.

    Counted pairwise, each training row against each test label window. Not
    against the fold's [min t, max t1] interval: that is fine for a contiguous
    block but meaningless for a shuffled one, where the interval spans nearly
    the whole sample and almost everything counts as overlapping.

    Returns (train_fraction, test_fraction) averaged over folds.
    """
    starts = pd.DatetimeIndex(t1.index).to_numpy()
    ends = pd.DatetimeIndex(t1.to_numpy()).to_numpy()
    train_fracs, test_fracs = [], []

    for train, test in cv.split():
        # (train x test) overlap on label windows.
        hit = ((ends[train][:, None] >= starts[test][None, :])
               & (starts[train][:, None] <= ends[test][None, :]))
        train_fracs.append(hit.any(axis=1).mean())
        test_fracs.append(hit.any(axis=0).mean())

    return float(np.mean(train_fracs)), float(np.mean(test_fracs))
