"""The design matrix, and preprocessing fitted inside folds.

The features are computed globally by `src/data_process.py`, which is not a
leak: each is a deterministic rolling function of its own past, shifted one
bar, so a row's values do not change with the fold it lands in.

The preprocessing is different. A scaler fitted on the whole sample lets every
fold's test rows contribute to the numbers used to transform them. The pipeline
still runs perfectly, which is what makes it subtle. `make_pipeline` returns an
estimator, so the only way to use it is to fit inside a fold.
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

#: Withheld, and spelt out because "everything except label and t1" would
#: quietly admit all of it.
#:
#: The raw bars are unshifted values at t, and a model that sees today's close
#: sees the direction of the move it is about to be scored on. `log_return` is
#: the return at t; `ret_1d` is the same number shifted and is the one that
#: belongs. `sigma` sets the barrier width, so handing it over would tell the
#: model how far price must travel for the label to fire -- `realized_vol_20d`
#: is its shifted version. The barriers come from sigma and the close at t.
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

    Raises on a missing feature rather than silently dropping it. The feature
    set is a configuration, so it has to be the list above rather than whatever
    the file happens to hold.
    """
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"labelled frame is missing features: {missing}")

    X = df[FEATURES].copy()
    y = df["label"].astype(int)
    return X, y, df["t1"]


def pre_holdout(df, holdout_start, bar_index):
    """The rows we may train or cross-validate on.

    Purged, not just filtered by date. A December 2022 row whose label resolves
    in January 2023 had its outcome decided by holdout bars, and a plain date
    filter keeps about ten of those per ticker.
    """
    return df.loc[purged_holdout_mask(df["t1"], holdout_start, bar_index)]


def make_pipeline(model):
    """Impute, scale, then the model, as one estimator fitted per fold.

    A Pipeline rather than pre-transformed arrays, so the only way to use it is
    to fit on a fold's training rows. Pre-scaling the whole matrix and then
    splitting is the mistake this exists to make impossible.

    The imputer has nothing to do today, since `compute_labels` drops the
    warm-up rows and the labelled files have no NaNs. It stays because a wider
    lookback would bring them back, and a test asserts it is still a no-op.

    No feature selection: ten features against ~660 effective observations does
    not need it, and a selector is one more thing to fit in-fold.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", model),
    ])
