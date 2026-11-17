"""Predictions into positions, and positions into costed returns.

Vectorised, daily, long or flat. We do not short, since borrow costs are not
something we can model honestly.

A signal at the close of *t* opens on the next bar and holds until the label
resolves at `t1`. The model was trained to answer "does the upper barrier fall
first", so the backtest runs that trade rather than a fixed ten-day hold.

Overlapping signals do not stack: position is capped at one unit, so we are in
or out. Otherwise leverage would depend on signal density and turnover would
mean nothing.
"""

import numpy as np
import pandas as pd


def positions(signal, t1, bar_index):
    """Binary position series over `bar_index`.

    `signal` is indexed by the bar the prediction is made on and is truthy
    where the model says act; `t1` gives each of those rows its resolution
    bar. A signal on *t* holds over `(t, t1]`.

    Difference array, so this is O(bars) rather than O(signals x bars).
    """
    bar_index = pd.DatetimeIndex(bar_index)
    acted = pd.Series(signal).astype(bool)
    acted = acted[acted]
    if len(acted) == 0:
        return pd.Series(0.0, index=bar_index, name="position")

    opened = pd.DatetimeIndex(acted.index)
    closed = pd.DatetimeIndex(t1.loc[acted.index].to_numpy())

    # `bar_index` must be the trading calendar, not the surviving rows: a kept
    # row's t1 can land on a bar that was itself dropped.
    end = bar_index.searchsorted(closed)
    if (end >= len(bar_index)).any() or not (bar_index[end] == closed).all():
        raise ValueError("some t1 dates are not bars in bar_index; pass the "
                         "full trading calendar, not the labelled rows")
    start = bar_index.searchsorted(opened) + 1

    live = np.zeros(len(bar_index) + 1)
    ok = start <= end                      # a label resolving on t+1 still trades
    np.add.at(live, start[ok], 1)
    np.add.at(live, end[ok] + 1, -1)

    held = np.cumsum(live)[:len(bar_index)] > 0
    return pd.Series(held.astype(float), index=bar_index, name="position")


def backtest(position, log_return, cost_bps=0.0):
    """Daily net returns, equity and turnover for one position series.

    Costs are charged on *changes* in position at `cost_bps` each way. Cost is
    a parameter rather than a constant because a strategy that works at zero
    and not at five is not a strategy.

    Simple returns, not log: a cost is a multiplicative haircut in and out, and
    mixing the two is a quiet source of error.
    """
    simple = np.expm1(pd.Series(log_return).reindex(position.index).fillna(0.0))

    traded = position.diff().abs().fillna(position.abs())
    gross = position * simple
    net = gross - traded * (cost_bps / 1e4)

    return pd.DataFrame({
        "position": position,
        "traded": traded,
        "gross_return": gross,
        "net_return": net,
        "equity": (1.0 + net).cumprod(),
    })


def turnover(traded, periods=252):
    """Annualised turnover, counting each leg.

    We report it because two strategies with the same gross Sharpe are not the
    same strategy if one trades ten times as often.
    """
    return float(np.asarray(traded).mean() * periods)


def exposure(position):
    """Share of bars held. Sharpe at 10% exposure is a different claim from
    Sharpe at 100%, the same way precision needs its coverage."""
    return float(np.asarray(position).mean())
