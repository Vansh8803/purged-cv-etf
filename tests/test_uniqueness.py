import pandas as pd
import numpy as np
import pytest
from src.uniqueness import concurrency, average_uniqueness, effective_sample_size


def t1_from(pairs, periods=10):
    """Build a t1 Series from (start_bar, end_bar) integer pairs."""
    bars = pd.bdate_range('2026-10-01', periods=periods)
    starts = [bars[a] for a, _ in pairs]
    ends = [bars[b] for _, b in pairs]
    return pd.Series(ends, index=pd.DatetimeIndex(starts)), bars


def test_non_overlapping_windows_are_fully_unique():
    # [0,1] and [2,3] share no bar, so every bar has concurrency 1.
    t1, bars = t1_from([(0, 1), (2, 3)])
    u = average_uniqueness(t1, bars)

    assert np.allclose(u.values, 1.0)
    assert effective_sample_size(t1, bars) == pytest.approx(2.0)


def test_hand_worked_three_label_chain():
    # Windows [0,1], [1,2], [2,3].
    # concurrency: bar0=1, bar1=2, bar2=2, bar3=1
    # u0 = mean(1/1, 1/2) = 0.75
    # u1 = mean(1/2, 1/2) = 0.50
    # u2 = mean(1/2, 1/1) = 0.75
    t1, bars = t1_from([(0, 1), (1, 2), (2, 3)])

    c = concurrency(t1, bars)
    assert list(c.values[:4]) == [1, 2, 2, 1]

    u = average_uniqueness(t1, bars)
    assert np.allclose(u.values, [0.75, 0.50, 0.75])
    assert u.mean() == pytest.approx(2.0 / 3.0)
    assert effective_sample_size(t1, bars) == pytest.approx(2.0)


def test_single_label_is_unique():
    t1, bars = t1_from([(0, 5)])
    assert average_uniqueness(t1, bars).iloc[0] == pytest.approx(1.0)


def test_nested_windows():
    # [0,2] and [1,2]: bar0=1, bar1=2, bar2=2
    # u0 = mean(1, 1/2, 1/2) = 2/3 ; u1 = mean(1/2, 1/2) = 1/2
    t1, bars = t1_from([(0, 2), (1, 2)])
    u = average_uniqueness(t1, bars)

    assert u.iloc[0] == pytest.approx(2.0 / 3.0)
    assert u.iloc[1] == pytest.approx(0.5)


def test_uniqueness_falls_as_windows_lengthen():
    # Ten labels starting on consecutive bars. Longer windows overlap more, so
    # average uniqueness has to fall.
    short, bars = t1_from([(i, i + 1) for i in range(10)], periods=30)
    long_, bars = t1_from([(i, i + 8) for i in range(10)], periods=30)

    assert average_uniqueness(short, bars).mean() > average_uniqueness(long_, bars).mean()


def test_uniqueness_is_bounded():
    t1, bars = t1_from([(i, i + 5) for i in range(12)], periods=30)
    u = average_uniqueness(t1, bars)

    assert ((u > 0) & (u <= 1.0)).all()


def test_t1_past_the_last_label_is_handled():
    # The real data does this: the last labelled row resolves on a bar that is
    # itself too late to label. The window must still be measured over every
    # bar it spans, not just the bars that happen to start a label.
    # Windows [0,1] and [1,9]: bar0=1, bar1=2, bars 2-9 = 1
    # u0 = mean(1, 1/2) = 0.75 ; u1 = (1/2 + 8) / 9
    t1, bars = t1_from([(0, 1), (1, 9)], periods=10)
    u = average_uniqueness(t1, bars)

    assert u.iloc[0] == pytest.approx(0.75)
    assert u.iloc[1] == pytest.approx((0.5 + 8.0) / 9.0)


def test_bar_axis_must_contain_every_date():
    t1, bars = t1_from([(0, 5)], periods=10)
    with pytest.raises(ValueError):
        average_uniqueness(t1, bars[:3])   # axis too short to hold t1


def test_concurrency_cannot_exceed_window_length():
    # A window [t, t+h] spans h+1 bars, so at most h+1 windows cover any bar.
    # On the real data with horizon 10 this ceiling is hit exactly, which is a
    # cheap check that windows are not being double counted.
    horizon = 4
    t1, bars = t1_from([(i, i + horizon) for i in range(20)], periods=40)
    c = concurrency(t1, bars)

    assert c.max() == horizon + 1
