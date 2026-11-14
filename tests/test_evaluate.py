import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from src.evaluate import cross_val_scores, paired_gap, summarise


class ActOnFlag(BaseEstimator, ClassifierMixin):
    """Acts wherever the `flag` column is 1. Not fitted; fully predictable."""

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        return X["flag"].to_numpy().astype(int)

    def predict_proba(self, X):
        p = X["flag"].to_numpy().astype(float)
        return np.column_stack([1.0 - p, p])


class TwoFolds:
    """Two folds of deliberately unequal size, so pooling and averaging differ."""

    def __init__(self, small, large):
        self.small, self.large = small, large

    def split(self, X=None, y=None, groups=None):
        yield np.array([]), np.asarray(self.small)
        yield np.array([]), np.asarray(self.large)


def test_precision_is_pooled_not_averaged_over_folds():
    # Fold A: 2 actions, both right        -> fold precision 1.00
    # Fold B: 10 actions, 4 right          -> fold precision 0.40
    # Unweighted mean 0.70; pooled is 6/12 = 0.50. The pooled number is the
    # one that answers "when it acted, how often was it right".
    n = 20
    X = pd.DataFrame({"flag": 0}, index=range(n))
    y = pd.Series(0, index=range(n))

    X.loc[[0, 1], "flag"] = 1
    y.loc[[0, 1]] = 1                                   # 2 actions, 2 correct
    X.loc[range(10, 20), "flag"] = 1
    y.loc[range(10, 14)] = 1                            # 10 actions, 4 correct

    scores = cross_val_scores(ActOnFlag(), X, y, TwoFolds(range(0, 2), range(10, 20)))
    s = summarise(scores)

    assert list(scores["n_acts"]) == [2, 10]
    assert list(scores["n_true_pos"]) == [2, 4]
    assert s["precision"] == pytest.approx(6 / 12)
    assert s["precision_fold_mean"] == pytest.approx(0.70)
    assert s["precision"] < s["precision_fold_mean"]    # the inflation


def test_coverage_is_the_share_of_bars_acted_on():
    n = 20
    X = pd.DataFrame({"flag": 0}, index=range(n))
    X.loc[range(10, 20), "flag"] = 1
    y = pd.Series([0, 1] * 10, index=range(n))

    s = summarise(cross_val_scores(ActOnFlag(), X, y, TwoFolds(range(0, 2), range(10, 20))))

    # 0 actions in fold A, 10 in fold B, over 12 tested bars.
    assert s["coverage"] == pytest.approx(10 / 12)


def test_positive_rate_is_not_the_majority_rate():
    # The confusion that made every model look broken: at 40% positives the
    # precision benchmark is 0.40 and the majority rate is 0.60.
    n = 100
    X = pd.DataFrame({"flag": 1}, index=range(n))
    # Positives spread evenly, so each fold carries the same 40/60 mix. Piling
    # them into one fold would make the per-fold majority rate meaningless.
    y = pd.Series([1 if i % 5 < 2 else 0 for i in range(n)], index=range(n))

    s = summarise(cross_val_scores(ActOnFlag(), X, y, TwoFolds(range(50), range(50, 100))))

    assert s["positive_rate"] == pytest.approx(0.40)
    assert s["majority_rate"] == pytest.approx(0.60)
    # Always-act's precision IS the positive rate, which is why that is the
    # number to beat.
    assert s["precision"] == pytest.approx(s["positive_rate"])


def folds_frame(gaps):
    """Per-fold frame for two schemes whose AUC differs by `gaps` per fold."""
    rows = []
    for fold, g in enumerate(gaps):
        rows.append({"ticker": "X", "model": "m", "fold": fold,
                     "scheme": "control_kfold", "auc": 0.60 + g})
        rows.append({"ticker": "X", "model": "m", "fold": fold,
                     "scheme": "purged_kfold", "auc": 0.60})
    return pd.DataFrame(rows)


def test_paired_gap_uses_the_spread_of_differences():
    gaps = [0.01, 0.02, 0.03, 0.04, 0.05]
    out = paired_gap(folds_frame(gaps), "control_kfold", "purged_kfold")

    expected_se = np.std(gaps, ddof=1) / np.sqrt(len(gaps))
    assert out.loc["m", "gap"] == pytest.approx(np.mean(gaps))
    assert out.loc["m", "paired_se"] == pytest.approx(expected_se)
    assert out.loc["m", "n"] == 5
    assert out.loc["m", "t"] == pytest.approx(np.mean(gaps) / expected_se)


def test_paired_se_is_tighter_than_the_unpaired_one_when_folds_move_together():
    # Why pairing matters. Both schemes swing wildly fold to fold, but almost
    # together, so the difference is steady. Comparing the gap against either
    # scheme's own spread would call a well-pinned gap noise.
    rows = []
    for fold, base in enumerate([0.50, 0.70, 0.55, 0.80, 0.60]):
        rows.append({"ticker": "X", "model": "m", "fold": fold,
                     "scheme": "control_kfold", "auc": base + 0.01})
        rows.append({"ticker": "X", "model": "m", "fold": fold,
                     "scheme": "purged_kfold", "auc": base})
    f = pd.DataFrame(rows)

    paired_se = paired_gap(f, "control_kfold", "purged_kfold").loc["m", "paired_se"]
    unpaired = f[f.scheme == "purged_kfold"]["auc"]
    unpaired_se = unpaired.std(ddof=1) / np.sqrt(len(unpaired))

    assert paired_se == pytest.approx(0.0, abs=1e-12)
    assert unpaired_se > 0.04


def test_paired_gap_reports_nan_t_for_a_rule_that_cannot_differ():
    # A deterministic rule scores identically under every scheme, so its gap is
    # exactly zero with zero spread. That is degenerate, not significant.
    out = paired_gap(folds_frame([0.0] * 5), "control_kfold", "purged_kfold")

    assert out.loc["m", "gap"] == 0.0
    assert out.loc["m", "paired_se"] == 0.0
    assert np.isnan(out.loc["m", "t"])


def test_paired_gap_rejects_a_missing_scheme():
    with pytest.raises(ValueError, match="need both"):
        paired_gap(folds_frame([0.01]), "control_kfold", "walk_forward")
