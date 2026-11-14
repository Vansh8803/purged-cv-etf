"""Steps 8 and 9: Sharpe with its error bars, drawdown, and the deflated ratio.

The point of this module is that a Sharpe ratio on its own is not evidence.

Sharpe is a mean divided by a standard deviation, which makes it a t-statistic
in disguise, so it has a sampling error -- and on a few thousand daily bars
that error is wide enough to swallow most results anybody reports. Then, on top
of the sampling error, there is the selection problem: trying five models and
reporting the best is a multiple-testing procedure whether or not it felt like
one. The Deflated Sharpe Ratio handles the second; `sharpe_se` handles the
first; they are different problems and both are needed.

**Per-period, not annualised.** Everything in the DSR is computed on per-period
returns. Annualising multiplies Sharpe by sqrt(252) but leaves skew, kurtosis
and the trial distribution alone, so feeding an annualised ratio into the
deflation formula silently inflates it by an order of magnitude. `sharpe` takes
a `periods` argument and annualises; `deflated_sharpe` deliberately does not.
"""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

#: Euler-Mascheroni, from the expected-maximum formula in Bailey and
#: Lopez de Prado (2014).
EULER = 0.577215664901532860606512


def _clean(returns):
    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    if len(r) < 2:
        raise ValueError("need at least two returns")
    return r


def sharpe(returns, periods=252):
    """Annualised Sharpe. Zero risk-free rate, stated rather than assumed.

    Returns 0.0 for a return series that never moves, which is what a strategy
    that never trades produces -- a degenerate case, not an error.
    """
    r = _clean(returns)
    sd = r.std(ddof=1)
    return 0.0 if sd == 0 else float(r.mean() / sd * np.sqrt(periods))


def sharpe_se(returns, periods=252):
    """Standard error of the Sharpe ratio, Lo (2002), IID case.

        SE(SR) = sqrt( (1 + SR^2 / 2) / T )

    on the per-period ratio, then annualised with the ratio itself. The IID
    assumption is the weak point and we say so: daily returns are serially
    correlated and volatility clusters, so this understates the true error.
    It is still far better than the usual practice of quoting no error at all.
    """
    r = _clean(returns)
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    sr = r.mean() / sd
    return float(np.sqrt((1.0 + 0.5 * sr ** 2) / len(r)) * np.sqrt(periods))


def max_drawdown(returns):
    """Worst peak-to-trough fall of the compounded equity curve, as a positive
    fraction. A good Sharpe with a 60% drawdown is not something anyone holds.
    """
    equity = (1.0 + _clean(returns)).cumprod()
    return float((1.0 - equity / np.maximum.accumulate(equity)).max())


def expected_max_sharpe(n_trials, sr_std):
    """E[max SR] across `n_trials` independent trials of zero true skill.

    The benchmark the observed ratio has to clear. Bailey and Lopez de Prado
    (2014), equation 5:

        E[max] ~ sr_std * ( (1 - g) Z^-1(1 - 1/N) + g Z^-1(1 - 1/(N e)) )

    with g the Euler-Mascheroni constant. It grows with both the number of
    things tried and how much they varied: twenty near-identical configurations
    buy far less luck than twenty wildly different ones, which is why the
    spread of the trials and not just their count goes into the formula.

    Per-period, like everything the DSR touches.
    """
    if n_trials < 2 or sr_std <= 0:
        return 0.0
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sr_std * ((1.0 - EULER) * z1 + EULER * z2))


def deflated_sharpe(returns, n_trials, sr_std):
    """Probability the observed Sharpe reflects skill rather than the best of N.

    Bailey and Lopez de Prado (2014). Takes the observed per-period ratio, the
    benchmark `expected_max_sharpe` above, and the shape of the return series,
    and returns a probability:

        DSR = Z[ (SR - SR*) sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2) ]

    **Why skew and kurtosis are in there.** The deflation asks how good the
    luckiest of N tries would look if none of them had any skill, and that
    depends on the shape of the distribution being drawn from. Returns are
    negatively skewed and fat-tailed, so assuming normality understates how
    extreme the best of N can get and the correction comes out too weak. `g4`
    is raw kurtosis -- 3 for a normal -- not the excess form.

    **The catch, which belongs in the write-up.** `n_trials` is whatever we
    honestly admit to having tried. Undercount it and this number is worse than
    useless, because it looks rigorous. That is the entire reason the decision
    log keeps a trial counter.
    """
    r = _clean(returns)
    sd = r.std(ddof=1)
    if sd == 0:
        return float("nan")

    sr = r.mean() / sd
    g3 = float(skew(r))
    g4 = float(kurtosis(r, fisher=False))
    sr_star = expected_max_sharpe(n_trials, sr_std)

    var = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2
    if var <= 0:
        return float("nan")

    return float(norm.cdf((sr - sr_star) * np.sqrt(len(r) - 1.0) / np.sqrt(var)))


def summary(returns, traded=None, position=None, n_trials=None, sr_std=None,
            periods=252):
    """Everything step 8 reports for one strategy, plus the DSR when asked.

    `n_trials` and `sr_std` are optional because the DSR is a property of the
    whole search, not of one backtest: it needs the spread of Sharpes across
    every configuration tried, which only the caller running the comparison
    knows.
    """
    out = {
        "sharpe": sharpe(returns, periods),
        "sharpe_se": sharpe_se(returns, periods),
        "max_drawdown": max_drawdown(returns),
        "total_return": float((1.0 + _clean(returns)).prod() - 1.0),
    }
    out["sharpe_t"] = (out["sharpe"] / out["sharpe_se"]
                       if out["sharpe_se"] > 0 else np.nan)
    if traded is not None:
        from src.backtest import turnover
        out["turnover"] = turnover(traded, periods)
    if position is not None:
        from src.backtest import exposure
        out["exposure"] = exposure(position)
    if n_trials is not None and sr_std is not None:
        out["dsr"] = deflated_sharpe(returns, n_trials, sr_std)
        out["sr_star_annual"] = expected_max_sharpe(n_trials, sr_std) * np.sqrt(periods)
    return out
