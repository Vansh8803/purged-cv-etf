"""Step 5: the design matrix, and preprocessing that is fitted inside folds.

Two separate things live here and the distinction is the whole point of the
step.

The *features* themselves are already computed, globally, by
`src/data_process.py`. That is not a leak: every one of them is a deterministic
rolling function of its own past, shifted one bar, so a row's feature values do
not change depending on which fold the row lands in. Nothing is estimated from
the sample.

The *preprocessing* is different. An imputer learns column means and a scaler
learns column means and standard deviations, and if either is fitted on the
whole sample then every fold's test rows have contributed to the numbers used
to transform them. That is a real leak, it is subtle because the pipeline still
runs perfectly, and `make_pipeline` is what prevents it: the estimator is fitted
inside each fold by whoever calls `fit`, which for us is the splitter loop at
step 6.
"""

from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.purging import purged_holdout_mask

ROOT = Path(__file__).resolve().parents[1]

#: The model sees these and nothing else. All ten are shifted one bar by
#: data_process.py, so a row dated t holds values computed up to t-1.
FEATURES = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d",
    "realized_vol_20d", "vol_of_vol_20d",
    "dist_ma_20d", "dist_ma_50d", "dist_ma_200d",
    "vol_zscore_20d",
]

#: Present in the labelled files and deliberately withheld. Spelt out because
#: "everything except label and t1" would quietly admit all of these.
#:
#: - Open/High/Low/Close/Adj Close/Volume, adj_high, adj_low: raw bar values at
#:   t, unshifted. A model that can see today's close can see the direction of
#:   today's move, which the label is about to score it on.
#: - log_return: the return *at* t, unshifted. `ret_1d` is the same quantity
#:   shifted, and is the one that belongs in the matrix.
#: - sigma: sigma_t, unshifted, and it sets the barrier width. Handing it over
#:   would both break the one-bar rule and tell the model how far price has to
#:   travel for the label to fire. `realized_vol_20d` is sigma shifted, and is
#:   the honest version.
#: - upper_barrier/lower_barrier: derived from sigma_t and the close at t.
EXCLUDED = [
    "Open", "High", "Low", "Close", "Adj Close", "Volume",
    "adj_high", "adj_low", "log_return",
    "sigma", "upper_barrier", "lower_barrier",
]


def load_labelled(ticker, processed_dir=None):
    """Read one ticker's labelled file, as written by scripts/build_labels.py."""
    processed_dir = Path(processed_dir or ROOT / "data" / "processed")
    df = pd.read_csv(processed_dir / f"{ticker}_labelled.csv",
                     index_col="Date", parse_dates=True)
    df["t1"] = pd.to_datetime(df["t1"])
    return df


def design_matrix(df):
    """Split a labelled frame into (X, y, t1).

    Raises on a missing or unexpected feature column rather than silently
    dropping or admitting one: the feature set is a configuration, so it has to
    be the list above and not whatever the file happens to contain.
    """
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"labelled frame is missing features: {missing}")

    X = df[FEATURES].copy()
    y = df["label"].astype(int)
    return X, y, df["t1"]


def pre_holdout(df, holdout_start, bar_index):
    """The rows that may be trained or cross-validated on.

    Purged, not merely filtered by date. A row dated December 2022 whose label
    resolves in January 2023 had its outcome decided by holdout bars, so a plain
    `df.index < holdout_start` keeps about ten rows per ticker whose labels came
    out of the holdout. This is the same `purge_mask` as everywhere else, with
    the holdout as the test block.
    """
    return df.loc[purged_holdout_mask(df["t1"], holdout_start, bar_index)]


def make_pipeline(model):
    """Impute, scale, then the model -- as one estimator, fitted per fold.

    Returned as a Pipeline rather than as pre-transformed arrays so that the
    only way to use it is to call `fit` on a fold's training rows. Pre-scaling
    the whole matrix and then splitting is the mistake this step exists to make
    impossible, and it is invisible once made.

    The imputer currently has nothing to do -- `compute_labels` drops the
    warm-up rows, so the labelled files have no NaNs at all. It stays because a
    wider lookback or an added feature would reintroduce them, and discovering
    that through a crash inside a fold is better than through a silently
    dropped row. The test suite asserts it is a no-op today, so if that ever
    changes it is noticed rather than absorbed.

    No feature selection. Ten features against an effective sample of roughly
    700 does not need it, and a selector is one more thing to fit in-fold and
    one more configuration to count.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", model),
    ])
