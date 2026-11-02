import pandas as pd
import numpy as np
import pytest
from src.cv import PurgedKFold


def labels(n=120, horizon=10, periods=160):
    """`n` labels on consecutive bars, each resolving `horizon` bars later."""
    bars = pd.bdate_range(start='2026-01-05', periods=periods)
    t1 = pd.Series([bars[i + horizon] for i in range(n)], index=bars[:n])
    return t1, bars


def test_folds_partition_the_rows_in_time_order():
    t1, bars = labels()
    folds = [te for _, te in PurgedKFold(t1, bars, n_splits=5).split()]

    assert sorted(np.concatenate(folds)) == list(range(len(t1)))
    assert all((np.diff(te) == 1).all() for te in folds)          # contiguous
    assert all(folds[i][-1] < folds[i + 1][0] for i in range(4))  # in order


def test_no_surviving_training_row_overlaps_its_test_fold():
    # The property the whole step exists for, checked directly against the
    # definition rather than against the implementation's own arithmetic.
    t1, bars = labels()
    starts, ends = t1.index, pd.DatetimeIndex(t1.to_numpy())

    for embargo in (0, 1, 5, 20):
        for train, test in PurgedKFold(t1, bars, 5, embargo).split():
            lo, hi = starts[test].min(), ends[test].max()
            overlaps = (ends[train] >= lo) & (starts[train] <= hi)

            assert not overlaps.any()
            assert len(np.intersect1d(train, test)) == 0


def test_purging_removes_rows_a_naive_split_would_keep():
    # If this ever comes out equal, the splitter is not purging anything.
    t1, bars = labels()
    purged = sum(len(tr) for tr, _ in PurgedKFold(t1, bars, 5).split())
    naive = sum(len(t1) - len(te) for _, te in PurgedKFold(t1, bars, 5).split())

    assert purged < naive


def test_training_set_shrinks_as_the_embargo_grows():
    t1, bars = labels()
    sizes = [sum(len(tr) for tr, _ in PurgedKFold(t1, bars, 5, e).split())
             for e in (0, 5, 20)]

    assert sizes[0] > sizes[1] > sizes[2]


def test_embargoed_rows_all_sit_just_after_their_fold():
    # Whatever the embargo removes beyond the purge must start after the fold's
    # span and within `embargo` bars of it -- never before, never further.
    t1, bars = labels()
    embargo = 7

    for (base, test), (tight, _) in zip(PurgedKFold(t1, bars, 5).split(),
                                        PurgedKFold(t1, bars, 5, embargo).split()):
        removed = t1.index[np.setdiff1d(base, tight)]
        if len(removed) == 0:
            continue
        end = t1.iloc[test].max()
        limit = bars[min(bars.searchsorted(end) + embargo, len(bars) - 1)]

        assert (removed > end).all()
        assert (removed <= limit).all()


def test_get_n_splits_and_dataframe_input():
    t1, bars = labels()
    cv = PurgedKFold(t1, bars, n_splits=4)
    X = pd.DataFrame(np.zeros((len(t1), 3)), index=t1.index)

    assert cv.get_n_splits() == 4
    assert len(list(cv.split(X))) == 4


def test_misaligned_x_raises():
    # A silent misalignment here would corrupt every fold while looking healthy,
    # so it has to fail loudly rather than split on whatever lines up.
    t1, bars = labels()
    with pytest.raises(ValueError):
        list(PurgedKFold(t1, bars).split(np.zeros((7, 2))))


def test_constructor_guards():
    t1, bars = labels()

    with pytest.raises(ValueError):
        PurgedKFold(t1, bars, n_splits=1)
    with pytest.raises(ValueError):
        PurgedKFold(t1, bars, embargo=-1)
    with pytest.raises(ValueError):
        PurgedKFold(t1.iloc[::-1], bars)                 # unsorted
    with pytest.raises(ValueError):
        PurgedKFold(pd.concat([t1, t1]).sort_index(), bars)   # duplicate dates


def test_more_splits_than_rows_raises():
    t1, bars = labels(n=4)
    with pytest.raises(ValueError):
        list(PurgedKFold(t1, bars, n_splits=5).split())
