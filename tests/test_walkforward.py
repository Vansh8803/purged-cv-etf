import pandas as pd
import numpy as np
import pytest
from src.cv import WalkForward


def labels(n=120, horizon=10, periods=160):
    """`n` labels on consecutive bars, each resolving `horizon` bars later."""
    bars = pd.bdate_range(start='2026-01-05', periods=periods)
    t1 = pd.Series([bars[i + horizon] for i in range(n)], index=bars[:n])
    return t1, bars


def test_never_trains_on_the_future():
    # The property the scheme exists for. Every training row must sit strictly
    # before the first test row, checked on dates rather than positions.
    t1, bars = labels()

    for train, test in WalkForward(t1, bars, n_splits=5).split():
        assert t1.index[train].max() < t1.index[test].min()


def test_training_set_expands():
    t1, bars = labels()
    sizes = [len(tr) for tr, _ in WalkForward(t1, bars, n_splits=5).split()]

    assert sizes == sorted(sizes)
    assert sizes[0] < sizes[-1]


def test_test_blocks_are_contiguous_ordered_and_skip_the_first():
    t1, bars = labels()
    folds = [te for _, te in WalkForward(t1, bars, n_splits=5).split()]

    assert all((np.diff(te) == 1).all() for te in folds)
    assert all(folds[i][-1] < folds[i + 1][0] for i in range(4))
    # One more block than folds, and the first is only ever trained on.
    assert folds[0][0] > 0
    assert np.concatenate(folds)[-1] == len(t1) - 1


def test_no_surviving_training_row_overlaps_its_test_block():
    # Training on the past is not enough on its own: the last rows before a
    # block resolve into it. This is what purging removes.
    t1, bars = labels()
    starts, ends = t1.index, pd.DatetimeIndex(t1.to_numpy())

    for train, test in WalkForward(t1, bars, n_splits=5).split():
        lo, hi = starts[test].min(), ends[test].max()
        overlaps = (ends[train] >= lo) & (starts[train] <= hi)

        assert not overlaps.any()
        assert len(np.intersect1d(train, test)) == 0


def test_purging_removes_the_boundary_rows():
    # Without purging, fold 0 would train on its whole first block. The rows
    # whose windows reach into the test block are exactly what goes.
    t1, bars = labels(horizon=10)
    train, test = next(iter(WalkForward(t1, bars, n_splits=5).split()))

    assert len(train) == test[0] - 10


def test_embargo_is_a_no_op_on_an_expanding_window():
    # Pinned as a property, not an omission: the embargo drops rows starting
    # after the test block, and an expanding window has none. Walk-forward is
    # therefore flat across the embargo sweep, which the write-up must explain.
    t1, bars = labels()
    base = [tr for tr, _ in WalkForward(t1, bars, 5, embargo=0).split()]

    for embargo in (1, 5, 20):
        other = [tr for tr, _ in WalkForward(t1, bars, 5, embargo=embargo).split()]
        assert all(np.array_equal(a, b) for a, b in zip(base, other))


def test_get_n_splits_and_dataframe_input():
    t1, bars = labels()
    cv = WalkForward(t1, bars, n_splits=4)
    X = pd.DataFrame({'f': np.arange(len(t1))}, index=t1.index)

    assert cv.get_n_splits() == 4
    assert len(list(cv.split(X))) == 4


def test_misaligned_x_raises():
    t1, bars = labels()
    with pytest.raises(ValueError, match="rows"):
        list(WalkForward(t1, bars).split(pd.DataFrame({'f': [1, 2, 3]})))


def test_constructor_guards():
    t1, bars = labels()
    with pytest.raises(ValueError, match="at least 1"):
        WalkForward(t1, bars, n_splits=0)
    with pytest.raises(ValueError, match="negative"):
        WalkForward(t1, bars, embargo=-1)
    with pytest.raises(ValueError, match="sorted"):
        WalkForward(t1.iloc[::-1], bars)
    with pytest.raises(ValueError, match="duplicate"):
        WalkForward(pd.concat([t1, t1.iloc[:1]]).sort_index(), bars)


def test_too_few_rows_raises():
    t1, bars = labels(n=3)
    with pytest.raises(ValueError, match="needs"):
        list(WalkForward(t1, bars, n_splits=5).split())
