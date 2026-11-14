"""Step 7: turning a classifier's output into a position series, and costing it.

Vectorised, at daily frequency, long or flat only. An event-driven backtester
would buy nothing here: at one bar per day there is no queue, no partial fill
and no intraday path to simulate, and the vectorised form is far easier to
audit line by line. The long-or-flat restriction was decided on 2026-10-14 --
shorting brings borrow costs we cannot model honestly.

**The holding period comes from the label, not from a fixed horizon.** A
prediction made at the close of day *t* opens a position on the next bar and
holds it until that label resolves on `t1`, which is the barrier touch or the
ten-bar timeout. That is deliberate: the model was trained to answer "does the
upper barrier fall first", so the trade the backtest runs has to be the trade
the label scores. Holding for a fixed ten days instead would measure something
the model was never asked about.

**Overlapping signals do not stack.** Labels overlap by construction -- that is
the entire subject of this project -- so on most bars several predictions are
live at once. Position is capped at one unit, so the strategy is "in or out"
rather than "in up to nine times", which keeps leverage constant and makes
turnover mean what it says. The cost of this is that a strong run of signals
looks the same as a single one; the benefit is that no number here depends on a
position-sizing rule nobody has justified.
"""

import numpy as np
import pandas as pd


def positions(signal, t1, bar_index):
    """Binary position series over `bar_index`, from act/don't-act predictions.

    `signal` is indexed by the bar the prediction is made on and is truthy where
    the model says act. `t1` is the matching resolution date for each of those
    rows.

    A signal on *t* holds the position over `(t, t1]` -- opened on the bar after
    the decision, closed on the bar the label resolves. Opening on *t* itself
    would be the classic lookahead: acting on a close at that same close.

    Built with a difference array so it stays O(bars) rather than O(signals x
    bars), and capped at one unit.
    """
    bar_index = pd.DatetimeIndex(bar_index)
    acted = pd.Series(signal).astype(bool)
    acted = acted[acted]
    if len(acted) == 0:
        return pd.Series(0.0, index=bar_index, name="position")

    opened = pd.DatetimeIndex(acted.index)
    closed = pd.DatetimeIndex(t1.loc[acted.index].to_numpy())

    # `bar_index` is the trading calendar, not the surviving rows. A kept row's
    # t1 can land on a bar that was itself purged, so passing the row index
    # instead silently shortens the calendar -- say so rather than run off it.
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
    """Daily net returns, equity curve and turnover for one position series.

    Costs are charged on *changes* in position, which is where they are
    actually incurred, at `cost_bps` basis points of the traded notional each
    way. A strategy that works at zero cost and not at five is not a strategy,
    which is why cost is a parameter here rather than a constant.

    Works in simple returns rather than log. Log returns add across time, which
    is why the features use them, but a cost is a multiplicative haircut on the
    way in and out, and mixing the two is a quiet source of error. The equity
    curve is the exact compounding of what the strategy earned.

    Returns a frame indexed by bar: position, net return, equity, and the
    traded amount on each bar.
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
    """Annualised round-trip turnover: how much the strategy trades.

    Reported because it is what the headline hides. Two strategies with the
    same Sharpe before costs are not the same strategy if one of them trades
    ten times as often, and at daily frequency turnover rather than signal
    quality usually decides whether anything survives the spread.
    """
    return float(np.asarray(traded).mean() * periods)


def exposure(position):
    """Share of bars actually held. Sharpe on 10% exposure is a different
    claim from Sharpe on 100%, the same way precision needs its coverage."""
    return float(np.asarray(position).mean())
