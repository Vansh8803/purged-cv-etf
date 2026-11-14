import numpy as np
import pandas as pd
import pytest

from src.backtest import backtest, exposure, positions, turnover


def bars(n=20):
    return pd.bdate_range('2026-01-05', periods=n)


def t1_from(idx, spans, cal):
    """t1 Series: label starting at bar i resolves at bar i + span."""
    return pd.Series([cal[i + s] for i, s in zip(idx, spans)],
                     index=cal[list(idx)])


def test_position_opens_the_bar_after_the_signal():
    # Acting on day t's close at day t's close is the classic lookahead. The
    # position starts at t+1 and that is the whole reason for this test.
    cal = bars()
    t1 = t1_from([2], [3], cal)
    pos = positions(pd.Series([True], index=[cal[2]]), t1, cal)

    assert pos.loc[cal[2]] == 0.0
    assert pos.loc[cal[3]] == 1.0


def test_position_closes_on_the_resolution_bar():
    # Held over (t, t1] -- inclusive at t1, because that is the bar the label
    # resolves on and the trade ends.
    cal = bars()
    t1 = t1_from([2], [3], cal)
    pos = positions(pd.Series([True], index=[cal[2]]), t1, cal)

    assert list(pos.loc[cal[3]:cal[5]]) == [1.0, 1.0, 1.0]
    assert pos.loc[cal[6]] == 0.0
    assert pos.sum() == 3.0


def test_overlapping_signals_do_not_stack():
    # Labels overlap by construction -- the subject of the whole project -- so
    # on most bars several predictions are live. Position stays at one unit.
    cal = bars()
    t1 = t1_from([2, 3, 4], [5, 5, 5], cal)
    pos = positions(pd.Series(True, index=cal[[2, 3, 4]]), t1, cal)

    assert pos.max() == 1.0
    assert set(np.unique(pos)) <= {0.0, 1.0}


def test_no_signal_means_no_position():
    cal = bars()
    t1 = t1_from([2], [3], cal)
    pos = positions(pd.Series([False], index=[cal[2]]), t1, cal)

    assert (pos == 0.0).all()


def test_t1_outside_the_calendar_raises():
    # Passing the surviving rows instead of the trading calendar silently
    # shortens it, and a kept row's t1 can land on a bar that was purged.
    cal = bars()
    t1 = t1_from([2], [5], cal)
    with pytest.raises(ValueError):
        positions(pd.Series([True], index=[cal[2]]), t1, cal[:4])


def test_gross_return_is_the_held_return():
    cal = bars(6)
    pos = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0, 0.0], index=cal)
    logret = pd.Series(np.log1p([0.01] * 6), index=cal)

    bt = backtest(pos, logret, cost_bps=0.0)
    assert bt["gross_return"].iloc[1] == pytest.approx(0.01)
    assert bt["gross_return"].iloc[3] == pytest.approx(0.0)
    assert bt["equity"].iloc[-1] == pytest.approx(1.01 ** 2)


def test_cost_is_charged_on_changes_only():
    # Two trades -- one in, one out -- at 10 bps each way.
    cal = bars(6)
    pos = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0, 0.0], index=cal)
    logret = pd.Series(0.0, index=cal)

    bt = backtest(pos, logret, cost_bps=10.0)
    assert bt["traded"].sum() == pytest.approx(2.0)
    assert bt["net_return"].sum() == pytest.approx(-2 * 10 / 1e4)


def test_more_cost_never_helps():
    cal = bars(30)
    rng = np.random.default_rng(0)
    pos = pd.Series(rng.integers(0, 2, 30).astype(float), index=cal)
    logret = pd.Series(rng.normal(0, 0.01, 30), index=cal)

    finals = [backtest(pos, logret, c)["equity"].iloc[-1] for c in (0, 5, 10)]
    assert finals[0] > finals[1] > finals[2]


def test_turnover_and_exposure_are_what_they_say():
    cal = bars(6)
    pos = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0, 0.0], index=cal)
    bt = backtest(pos, pd.Series(0.0, index=cal))

    assert exposure(bt["position"]) == pytest.approx(2 / 6)
    assert turnover(bt["traded"], periods=6) == pytest.approx(2.0)


def test_always_held_has_no_turnover_after_entry():
    cal = bars(10)
    pos = pd.Series(1.0, index=cal)
    bt = backtest(pos, pd.Series(0.0, index=cal))

    # One unit on the first bar to get in, nothing after.
    assert bt["traded"].iloc[1:].sum() == pytest.approx(0.0)
