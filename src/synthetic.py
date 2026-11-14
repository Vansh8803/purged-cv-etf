"""Step 10: the validity check. Does the pipeline find a signal we planted?

Without this a weak result on real data is ambiguous: it could mean there was no
edge, or it could mean our code never worked. Running the whole machinery on
data whose answer we already know separates those two, which is what turns a
null result into a finding rather than a failure.

The series goes through the *real* pipeline -- `engineer_features`,
`compute_barriers`, `compute_labels`, the real splitters, the real in-fold
pipeline. A shortcut that fed a model its features directly would test nothing,
because every bug worth catching lives in the parts it skipped.

One knob. `phi` is the AR(1) coefficient on daily returns: tomorrow's return is
`phi` times today's plus noise. At `phi = 0` the series is a pure random walk
and there is nothing to find. Above zero the predictability is real, observable,
and sits in a feature we already compute -- `ret_1d`, which is yesterday's
return -- so the model has an honest route to it and we know exactly what that
route is.
"""

import numpy as np
import pandas as pd

from src.data_process import engineer_features
from src.labels import compute_barriers, compute_labels


def synthetic_ohlc(n_bars=3000, phi=0.0, vol=0.01, intraday=0.6, seed=0):
    """Daily OHLCV whose returns follow an AR(1) with coefficient `phi`.

    `Adj Close` is set equal to `Close`, so the adjustment factor is exactly 1
    throughout, as it is for GLD. That keeps the high/low scaling path out of
    the way: this check is about whether a signal survives the evaluation
    machinery, and a dividend confound would muddy it for no gain.

    `intraday` scales the daily high-low range relative to `vol`. It matters
    more than it looks -- the range is what decides how quickly a barrier is
    touched, and therefore how long the label windows are and how much they
    overlap.
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


#: AUC the full feature set reaches on data containing no signal at all.
#:
#: Not 0.50. `sigma_t` sets the barrier width, wide barriers are less likely to
#: be touched within ten bars, an untouched barrier is a timeout, and a timeout
#: is label 0 -- so trailing volatility predicts the label on a pure random
#: walk. Measured at phi = 0 over ten seeds: 0.544 +/- 0.040, against 0.503 for
#: the return features alone. See decision-log.md, 2026-11-10.
#:
#: Real-data AUC is reported against this number rather than against 0.50.
#: `scripts/check_synthetic.py` recomputes it; if that run moves, update this.
NO_SIGNAL_FLOOR = 0.544
