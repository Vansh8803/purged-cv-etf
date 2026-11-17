"""Sharpe with its error bars, drawdown, turnover and the deflated ratio.

A Sharpe ratio on its own is not evidence. It is a mean over a standard
deviation, so it is a t-statistic in disguise and carries a sampling error wide
enough to swallow most results. On top of that, reporting the best of five
models is multiple testing whether or not it felt like it. `sharpe_se` handles
the first problem, `deflated_sharpe` the second.

Everything in the deflation is per-period. Annualising scales Sharpe by
sqrt(252) but leaves skew and kurtosis alone, so feeding it an annualised ratio
inflates the result silently.
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
    """Annualised Sharpe, zero risk-free rate.

    Returns 0.0 for a series that never moves, which is what a strategy that
    never trades gives us.
    """
    r = _clean(returns)
    sd = r.std(ddof=1)
    return 0.0 if sd == 0 else float(r.mean() / sd * np.sqrt(periods))


def sharpe_se(returns, periods=252):
    """Standard error of the Sharpe ratio, Lo (2002), IID case.

        SE(SR) = sqrt( (1 + SR^2 / 2) / T )

    computed on the per-period ratio, then annualised. The IID assumption is
    the weak point: daily returns are serially correlated and volatility
    clusters, so this understates the true error. Still better than quoting no
    error at all, which is the usual practice.
    """
    r = _clean(returns)
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    sr = r.mean() / sd
    return float(np.sqrt((1.0 + 0.5 * sr ** 2) / len(r)) * np.sqrt(periods))


def max_drawdown(returns):
    """Worst peak-to-trough fall of the equity curve, as a positive fraction.

    Sharpe assumes we held on through everything; this says whether anyone
    would have.
    """
    equity = (1.0 + _clean(returns)).cumprod()
    return float((1.0 - equity / np.maximum.accumulate(equity)).max())


def expected_max_sharpe(n_trials, sr_std):
    """E[max SR] across `n_trials` trials of zero skill. Per-period.

    Bailey and Lopez de Prado (2014), equation 5:

        E[max] ~ sr_std * ( (1 - g) Z^-1(1 - 1/N) + g Z^-1(1 - 1/(N e)) )

    with g the Euler-Mascheroni constant. It grows with both the count and the
    spread, since twenty near-identical configurations buy less luck than
    twenty different ones.
    """
    if n_trials < 2 or sr_std <= 0:
        return 0.0
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sr_std * ((1.0 - EULER) * z1 + EULER * z2))


def deflated_sharpe(returns, n_trials, sr_std):
    """Probability the observed Sharpe is skill rather than the best of N.

    Bailey and Lopez de Prado (2014):

        DSR = Z[ (SR - SR*) sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2) ]

    Skew and kurtosis are in there because the deflation asks how extreme the
    luckiest of N skill-less tries could look, which depends on the shape of
    the distribution being drawn from. `g4` is raw kurtosis, 3 for a normal.

    `n_trials` is whatever we honestly admit to trying. Undercount it and the
    number is worse than useless, because it still looks rigorous.
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
    """Everything we report for one strategy, plus the DSR when asked.

    `n_trials` and `sr_std` are optional because the DSR belongs to the whole
    search rather than to one backtest, so only the caller running the
    comparison knows them.
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


def exposure_matched_sharpe(position, returns, n_draws=200, seed=0, periods=252):
    """Sharpe from holding the same *number* of bars, picked at random.

    A strategy invested 22% of the time forfeits most of the market drift
    whatever it selects, so comparing its Sharpe against buy-and-hold confounds
    how well it chose with how often it was in. Shuffling the position series
    keeps the count and destroys which bars they were, so the gap between this
    and the real Sharpe is the selection ability.

    Buy-and-hold is the check: at ~100% exposure there is nothing to shuffle
    and the two agree.

    Returns (mean, sd) across draws, annualised like `sharpe`.
    """
    held = np.asarray(position, dtype=bool)
    r = np.asarray(pd.Series(returns).fillna(0.0), dtype=float)
    if len(held) != len(r):
        raise ValueError("position and returns must be the same length")

    rng = np.random.default_rng(seed)
    draws = np.empty(n_draws)
    for i in range(n_draws):
        perm = held.copy()
        rng.shuffle(perm)
        draws[i] = sharpe(np.where(perm, r, 0.0), periods)

    return float(draws.mean()), float(draws.std(ddof=1))
