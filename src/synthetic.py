"""Does the pipeline find a signal we planted ourselves?

Without this a weak result on real data is ambiguous: no edge, or broken code,
and we could not tell which. Running everything on data whose answer we already
know separates the two.

The series goes through the real pipeline rather than a shortcut, since every
bug worth catching lives in the parts a shortcut would skip.

One knob. `phi` is the AR(1) coefficient on daily returns. At 0 the series is a
random walk with nothing to find; above it the predictability lands in
`ret_1d`, so the model has a route we know about.
"""

import numpy as np
import pandas as pd

from src.data_process import engineer_features
from src.labels import compute_barriers, compute_labels


def synthetic_ohlc(n_bars=3000, phi=0.0, vol=0.01, intraday=0.6, seed=0):
    """Daily OHLCV whose returns follow an AR(1) with coefficient `phi`.

    `Adj Close` equals `Close`, so the adjustment factor is 1 throughout as it
    is for GLD, keeping the high/low scaling path out of the way.

    `intraday` scales the daily range against `vol`, and it matters more than
    it looks: the range decides how quickly a barrier is touched, so it sets
    how long the label windows are and how much they overlap.
    """
    rng = np.random.default_rng(seed)

    # AR(1) returns. The planted signal, and the only one present.
    eps = rng.normal(0.0, vol, n_bars)
    ret = np.zeros(n_bars)
    for i in range(1, n_bars):
        ret[i] = phi * ret[i - 1] + eps[i]

    close = 100.0 * np.exp(np.cumsum(ret))
    reach = np.abs(rng.normal(0.0, intraday * vol, n_bars))

    bars = pd.bdate_range("2005-01-03", periods=n_bars)
    return pd.DataFrame({
        "Open": close * np.exp(rng.normal(0.0, 0.2 * vol, n_bars)),
        "High": close * np.exp(reach),
        "Low": close * np.exp(-reach),
        "Close": close,
        "Adj Close": close,
        "Volume": rng.integers(1_000_000, 5_000_000, n_bars).astype(float),
    }, index=bars)


def synthetic_dataset(n_bars=3000, phi=0.0, barrier_mult=2.5, horizon=10,
                      vol_lookback=20, seed=0, **kwargs):
    """A labelled synthetic frame, built by the same code as the real ones."""
    raw = synthetic_ohlc(n_bars=n_bars, phi=phi, seed=seed, **kwargs)
    bar_index = raw.index

    df = engineer_features(raw)
    df = compute_barriers(df, barrier_mult=barrier_mult, vol_lookback=vol_lookback)
    df = df.dropna(subset=["sigma"])
    return compute_labels(df, horizon=horizon), bar_index


#: AUC the full feature set reaches on data with no signal in it at all.
#:
#: Not 0.50. `sigma_t` sets the barrier width, wide barriers are less likely to
#: be touched, an untouched barrier times out, and a timeout is label 0 -- so
#: trailing volatility predicts the label on a random walk. Measured at phi = 0
#: over ten seeds: 0.544 +/- 0.040, against 0.503 for the return features
#: alone. We report every real AUC against this rather than against 0.50.
#: `scripts/check_synthetic.py` recomputes it.
NO_SIGNAL_FLOOR = 0.544


def sign_flipped_ohlc(raw, seed=0):
    """Real bars with the direction of every day's move randomised.

    Each daily log return is multiplied by a random +/-1, which leaves `|r_t|`
    untouched bar for bar. The volatility path therefore survives -- clustering,
    fat tails, the 2008 and 2020 spikes -- while directional predictability is
    destroyed. On SPY the rolling 20-day sigma correlates 0.998 with the real
    one; not exact, because a window's variance depends on its mean too.

    Intraday reach carries over as magnitudes, so how readily a barrier is
    touched is preserved. That matters: the touch rate drives the timeout rate,
    and the timeout rate is the channel the floor measures.

    `Adj Close` equals `Close`, so the scaling path cannot confound the result.
    """
    af = raw["Adj Close"] / raw["Close"]
    close = raw["Adj Close"]
    up = np.abs(np.log((raw["High"] * af / close).to_numpy()))
    down = np.abs(np.log((raw["Low"] * af / close).to_numpy()))

    r = np.log(close / close.shift(1)).fillna(0.0).to_numpy()
    flip = np.random.default_rng(seed).choice([-1.0, 1.0], len(r))
    path = close.iloc[0] * np.exp(np.cumsum(flip * r))

    return pd.DataFrame({
        "Open": path,
        "High": path * np.exp(up),
        "Low": path * np.exp(-down),
        "Close": path,
        "Adj Close": path,
        "Volume": raw["Volume"].to_numpy(),
    }, index=raw.index)
