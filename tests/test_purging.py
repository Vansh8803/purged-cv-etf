import pandas as pd
import numpy as np
import pytest
from src.purging import (
    purge_mask, block_span, embargo_bars, purged_holdout_mask,
)


def toy(horizon=2, n_labels=9, periods=12, start='2026-01-05'):
    """Labels on consecutive bars, each resolving `horizon` bars later.

    2026-01-05 is a Monday, so bars 0-4 are Mon-Fri and bar 5 is the next
    Monday. Several tests below lean on that weekend sitting between bar 4 and
    bar 5.
    """
    bars = pd.bdate_range(start=start, periods=periods)
    t1 = pd.Series([bars[i + horizon] for i in range(n_labels)],
                   index=bars[:n_labels])
    return t1, bars


def kept(mask):
    return set(np.flatnonzero(mask.to_numpy()))


def test_hand_worked_purge():
    # Test block is rows 4 and 5. Their labels resolve on bars 6 and 7, so the
    # block's span is [bar 4, bar 7]. Drop row i when t1_i >= bar4 and t_i <= bar7:
    #   row 0 t1=bar2, row 1 t1=bar3  -> resolved before the block, keep
    #   rows 2,3                      -> resolve inside it, drop
    #   rows 4,5                      -> the test rows themselves, drop
    #   rows 6,7                      -> start inside the block's reach, drop
    #   row 8  t=bar8                 -> starts after it, keep
    t1, bars = toy()
    mask = purge_mask(t1, bars[4], t1.iloc[[4, 5]].max(), bars)

    assert kept(mask) == {0, 1, 8}


def test_span_reaches_past_the_last_test_row():
    # The bug this guards: using the last test ROW's date as the span's end.
    # Rows 6 and 7 start after row 5 but their windows overlap row 5's label,
    # so they have to be purged. Measured by row dates they would survive.
    t1, bars = toy()
    start, end = block_span(t1.iloc[[4, 5]])

    assert start == bars[4]
    assert end == bars[7]          # max t1, not bars[5]
    assert end > t1.index[5]

    by_row_date = purge_mask(t1, bars[4], bars[5], bars)
    assert {6, 7} <= kept(by_row_date)      # the leak, if you measure it wrong
    assert not {6, 7} & kept(purge_mask(t1, start, end, bars))


def test_overlap_is_inclusive_at_both_ends():
    # A label resolving exactly on the first test bar overlaps it. A label
    # starting exactly on the last bar of the span overlaps it too. Strict
    # inequalities either side would let both through.
    t1, bars = toy()

    assert 2 not in kept(purge_mask(t1, bars[4], bars[7], bars))   # t1 == test_start
    assert 7 not in kept(purge_mask(t1, bars[4], bars[7], bars))   # t == test_end


def test_embargo_is_exclusive_at_start_inclusive_at_end():
    t1, bars = toy()
    end = t1.iloc[[4, 5]].max()                 # bar 7

    # Row 8 starts one bar after the span and is the only row left to embargo.
    assert 8 in kept(purge_mask(t1, bars[4], end, bars, embargo=0))
    assert 8 not in kept(purge_mask(t1, bars[4], end, bars, embargo=1))


def test_embargo_counts_bars_not_calendar_days():
    # The bug this guards: an embargo of 1 implemented as t + 1 DAY. Bar 4 is a
    # Friday, so one calendar day lands on the Saturday, which is not a bar,
    # and the Monday row survives when it should not.
    t1, bars = toy()
    assert bars[4].day_name() == 'Friday'
    assert bars[5].day_name() == 'Monday'
    assert (bars[5] - bars[4]).days == 3

    # Row 5 starts the Monday. At embargo 0 nothing touches it, because its
    # window does not reach back to bar 2.
    mask0 = purge_mask(t1, bars[0], bars[4], bars, embargo=0)
    mask1 = purge_mask(t1, bars[0], bars[4], bars, embargo=1)

    assert 5 in kept(mask0)
    assert 5 not in kept(mask1)


def test_embargo_is_clipped_at_the_end_of_the_sample():
    # A test block at the very end has nothing after it to embargo. Asking for
    # 20 bars of embargo must not run off the calendar.
    t1, bars = toy()
    mask = purge_mask(t1, bars[6], bars[11], bars, embargo=20)

    assert kept(mask) == kept(purge_mask(t1, bars[6], bars[11], bars, embargo=0))


def test_longer_embargo_never_keeps_more():
    t1, bars = toy(horizon=2, n_labels=9, periods=20)
    sizes = [purge_mask(t1, bars[3], bars[5], bars, embargo=e).sum()
             for e in range(0, 6)]

    assert sizes == sorted(sizes, reverse=True)


def test_embargo_without_bar_index_raises():
    # Silently ignoring the embargo would be the dangerous failure here.
    t1, bars = toy()
    with pytest.raises(ValueError):
        purge_mask(t1, bars[4], bars[7], None, embargo=1)


def test_test_end_must_be_a_bar():
    t1, bars = toy()
    saturday = pd.Timestamp('2026-01-10')
    assert saturday not in bars
    with pytest.raises(ValueError):
        purge_mask(t1, bars[4], saturday, bars, embargo=1)


def test_nat_in_t1_raises():
    # compute_labels drops unresolved rows, so a NaT here means something
    # upstream reintroduced them. Comparisons against NaT are False, which
    # would quietly leave such a row in every training set.
    t1, bars = toy()
    t1 = t1.copy()
    t1.iloc[3] = pd.NaT
    with pytest.raises(ValueError):
        purge_mask(t1, bars[4], bars[7], bars)


def test_reversed_span_raises():
    t1, bars = toy()
    with pytest.raises(ValueError):
        purge_mask(t1, bars[7], bars[4], bars)


def test_empty_test_block_raises():
    t1, _ = toy()
    with pytest.raises(ValueError):
        block_span(t1.iloc[[]])


def test_embargo_bars_from_fraction():
    assert embargo_bars(4302, 0.01) == 43
    assert embargo_bars(4302, 0.0) == 0
    assert embargo_bars(1000, 0.02) == 20


def test_holdout_purge_drops_rows_resolving_into_the_holdout():
    # The boundary case: a row dated before the holdout whose label resolves
    # inside it. Its label is a fact about holdout prices, so training on it
    # leaks the holdout however carefully the folds are purged.
    t1, bars = toy(horizon=2, n_labels=9, periods=12)
    holdout_start = bars[6]

    mask = purged_holdout_mask(t1, holdout_start, bars)
    survivors = t1[mask.to_numpy()]

    assert (survivors.index < holdout_start).all()
    assert (survivors < holdout_start).all()
    # Rows 4 and 5 resolve on bars 6 and 7, both inside the holdout.
    assert not {4, 5} & kept(mask)
    assert {0, 1, 2, 3} == kept(mask)


def test_bar_at_test_end_is_purged_whatever_the_embargo():
    # The embargo is documented as exclusive at test_end. That is a statement
    # about not double counting rather than observable behaviour: a row starting
    # exactly on test_end is already purged by the overlap rule, so flipping the
    # embargo's lower bound to inclusive cannot change any result. Mutating
    # `starts > test_end` to `>=` leaves this suite green, and that is correct
    # rather than a gap -- this test records why.
    t1, bars = toy()
    end = t1.iloc[[4, 5]].max()                 # bar 7, and row 7 starts there

    for embargo in (0, 1, 5):
        assert 7 not in kept(purge_mask(t1, bars[4], end, bars, embargo=embargo))


def test_block_span_does_not_assume_a_sorted_block():
    # PurgedKFold hands over sorted contiguous blocks, so this is latent rather
    # than live. It stops being latent the moment anything passes a scattered
    # block -- a shuffled control is exactly that -- and a span taken from
    # whichever row came first would start after some of its own test rows.
    t1, bars = toy()
    scattered = t1.iloc[[7, 2, 5]]

    assert block_span(scattered) == (bars[2], t1.iloc[7])
    assert block_span(scattered) == block_span(scattered.sort_index())
