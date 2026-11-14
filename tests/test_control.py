import pandas as pd
import numpy as np
import pytest
from src.cv import NaiveKFold, PurgedKFold, WalkForward, leakage


def labels(n=120, horizon=10, periods=160):
    bars = pd.bdate_range(start='2026-01-05', periods=periods)
    t1 = pd.Series([bars[i + horizon] for i in range(n)], index=bars[:n])
    return t1, bars


def test_control_trains_on_everything_outside_the_fold():
    # The omission being measured: no purge, no embargo, nothing held back.
    t1, bars = labels()

    for train, test in NaiveKFold(t1, bars, 5).split():
        assert len(train) + len(test) == len(t1)
        assert len(np.intersect1d(train, test)) == 0


def test_control_folds_partition_the_rows():
    t1, bars = labels()
    for shuffle in (False, True):
        folds = [te for _, te in NaiveKFold(t1, bars, 5, shuffle, seed=0).split()]
        assert sorted(np.concatenate(folds)) == list(range(len(t1)))


def test_contiguous_is_the_default():
    t1, bars = labels()
    folds = [te for _, te in NaiveKFold(t1, bars, 5).split()]

    assert all((np.diff(te) == 1).all() for te in folds)
    assert all(folds[i][-1] < folds[i + 1][0] for i in range(4))


def test_shuffling_scatters_the_folds():
    t1, bars = labels()
    folds = [te for _, te in NaiveKFold(t1, bars, 5, shuffle=True, seed=0).split()]

    # Test positions come back sorted, but no longer consecutive, and each
    # fold now spans nearly the whole sample instead of a block of it.
    assert all((np.diff(te) > 0).all() for te in folds)
    assert not any((np.diff(te) == 1).all() for te in folds)
    # Each shuffled fold reaches across most of the sample; a contiguous one
    # spans only its own fifth. Compared directly, so there is no magic number.
    contiguous = [te for _, te in NaiveKFold(t1, bars, 5).split()]
    spread = min(te.max() - te.min() for te in folds)
    assert spread > 3 * max(te.max() - te.min() for te in contiguous)


def test_shuffle_is_reproducible_from_the_seed():
    t1, bars = labels()
    a = [te for _, te in NaiveKFold(t1, bars, 5, shuffle=True, seed=7).split()]
    b = [te for _, te in NaiveKFold(t1, bars, 5, shuffle=True, seed=7).split()]
    c = [te for _, te in NaiveKFold(t1, bars, 5, shuffle=True, seed=8).split()]

    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))


def test_corrected_schemes_leak_nothing_and_the_control_leaks():
    # The headline comparison, as a property rather than a measured constant.
    t1, bars = labels()

    assert leakage(PurgedKFold(t1, bars, 5, 5), t1) == (0.0, 0.0)
    assert leakage(WalkForward(t1, bars, 5), t1) == (0.0, 0.0)

    contiguous, _ = leakage(NaiveKFold(t1, bars, 5), t1)
    shuffled, _ = leakage(NaiveKFold(t1, bars, 5, shuffle=True, seed=0), t1)

    assert 0 < contiguous < shuffled


def test_shuffling_leaks_far_more_than_fold_geometry_alone():
    # The point of keeping both: same code, same data, same scheme name, and
    # the leak differs by orders of magnitude purely from fold geometry.
    t1, bars = labels(n=400, horizon=10, periods=450)

    contiguous, _ = leakage(NaiveKFold(t1, bars, 5), t1)
    shuffled, _ = leakage(NaiveKFold(t1, bars, 5, shuffle=True, seed=0), t1)

    assert contiguous < 0.10
    assert shuffled > 0.80


def test_leakage_counts_pairwise_not_against_the_fold_interval():
    # Guards the bug that produced the bad 99.9% figure. One label far from a
    # tight cluster: measured pairwise it overlaps nothing, but against the
    # fold's [min t, max t1] interval it would look like an overlap.
    bars = pd.bdate_range('2026-01-05', periods=200)
    t1 = pd.Series(
        [bars[1], bars[2], bars[3], bars[150]],
        index=[bars[0], bars[1], bars[2], bars[149]],
    )

    class TwoFolds:
        def split(self):
            yield np.array([3]), np.array([0, 1])   # the lone late label

    train_frac, _ = leakage(TwoFolds(), t1)
    assert train_frac == 0.0


def test_control_guards():
    t1, bars = labels()
    with pytest.raises(ValueError, match="at least 2"):
        NaiveKFold(t1, bars, n_splits=1)
    with pytest.raises(ValueError, match="exceeds"):
        list(NaiveKFold(t1, bars, n_splits=len(t1) + 1).split())
    with pytest.raises(ValueError, match="rows"):
        list(NaiveKFold(t1, bars).split(pd.DataFrame({'f': [1, 2]})))
