import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src.features import FEATURES
from src.models import (
    AlwaysAct, MACrossover, MetaLabel, crossover_signal, model_zoo,
)


def frame(ma20, ma50, close=100.0, n=None):
    """A design matrix whose MA distances are derived from stated MA levels.

    Built from the moving averages rather than from the distances, so the test
    states what it means -- "the 20-day sits above the 50-day" -- and the
    algebra under test has to agree with it.
    """
    ma20, ma50 = np.asarray(ma20, float), np.asarray(ma50, float)
    n = len(ma20)
    X = pd.DataFrame(0.0, index=pd.bdate_range('2026-01-05', periods=n),
                     columns=FEATURES)
    X["dist_ma_20d"] = (close - ma20) / ma20
    X["dist_ma_50d"] = (close - ma50) / ma50
    return X


def test_crossover_signal_fires_when_short_ma_is_above_long():
    # The inequality on the distances runs the opposite way to the one on the
    # averages, so this is the test that stops the rule being a death cross.
    X = frame(ma20=[99.0, 98.0, 100.0], ma50=[98.0, 99.0, 100.0])

    assert list(crossover_signal(X)) == [True, False, False]


def test_crossover_signal_matches_the_moving_averages_at_random():
    rng = np.random.default_rng(0)
    ma20, ma50 = rng.uniform(50, 150, 400), rng.uniform(50, 150, 400)
    X = frame(ma20, ma50)

    assert np.array_equal(crossover_signal(X), ma20 > ma50)


def test_always_act_acts_every_day():
    X = frame(ma20=[99.0] * 5, ma50=[98.0] * 5)
    model = AlwaysAct().fit(X, pd.Series([0, 1, 0, 1, 1]))

    assert (model.predict(X) == 1).all()
    assert np.allclose(model.predict_proba(X)[:, 1], 1.0)


def test_always_act_precision_is_the_positive_rate():
    # Which is exactly the number every other model's precision is read against.
    X = frame(ma20=[99.0] * 10, ma50=[98.0] * 10)
    y = pd.Series([1] * 4 + [0] * 6, index=X.index)
    from sklearn.metrics import precision_score

    pred = AlwaysAct().fit(X, y).predict(X)
    assert precision_score(y, pred) == pytest.approx(y.mean())


def test_ma_crossover_follows_the_rule():
    X = frame(ma20=[99.0, 98.0, 101.0, 97.0], ma50=[98.0, 99.0, 100.0, 99.0])
    pred = MACrossover().fit(X, pd.Series([0, 0, 0, 0])).predict(X)

    assert list(pred) == [1, 0, 1, 0]


def test_rules_ignore_what_they_are_trained_on():
    # A rule cannot leak or overfit, which is what makes it the reference point:
    # any control-vs-purged difference a learned model shows is the learning.
    X = frame(ma20=[99.0, 98.0, 101.0], ma50=[98.0, 99.0, 100.0])
    truth = pd.Series([1, 0, 1], index=X.index)
    lies = pd.Series([0, 1, 0], index=X.index)

    assert np.array_equal(MACrossover().fit(X, truth).predict(X),
                          MACrossover().fit(X, lies).predict(X))


def meta_fixture(n=200, seed=0):
    rng = np.random.default_rng(seed)
    ma20 = rng.uniform(90, 110, n)
    X = frame(ma20, ma50=np.full(n, 100.0))
    X["ret_1d"] = rng.normal(size=n)
    y = pd.Series((rng.random(n) < 0.4).astype(int), index=X.index)
    return X, y


def test_meta_label_never_acts_where_the_rule_does_not():
    # The construction's whole point: it filters the rule's signals, it does not
    # generate its own.
    X, y = meta_fixture()
    model = model_zoo(0)["meta_crossover_logistic"].fit(X, y)
    pred = model.predict(X)

    assert not pred[~crossover_signal(X)].any()
    assert np.allclose(model.predict_proba(X)[~crossover_signal(X), 1], 0.0)


def test_meta_label_acts_when_there_is_something_to_learn():
    # Needs a learnable outcome. On noise the inner logistic predicts the
    # majority class everywhere and the meta model correctly acts nowhere --
    # which is the right behaviour, and why this fixture plants a signal rather
    # than reusing the random one.
    X, _ = meta_fixture()
    y = pd.Series((X["ret_1d"] > 0).astype(int).to_numpy(), index=X.index)
    model = model_zoo(0)["meta_crossover_logistic"].fit(X, y)
    pred = model.predict(X)

    assert pred.any()
    assert not pred[~crossover_signal(X)].any()


def test_meta_label_survives_a_fold_the_rule_never_fires_in():
    # A real fold can contain no crossover at all, or only one class of outcome.
    # Predicting nothing is the honest answer; crashing inside fold three of
    # fifteen tickers is not.
    X = frame(ma20=[98.0] * 50, ma50=[99.0] * 50)       # rule never fires
    y = pd.Series([0, 1] * 25, index=X.index)
    model = MetaLabel(model_zoo(0)["logistic"]).fit(X, y)

    assert model.fitted_ is False
    assert not model.predict(X).any()
    assert model.predict_proba(X).shape == (50, 2)


def test_meta_label_declines_a_single_class_fold():
    X = frame(ma20=[101.0] * 50, ma50=[99.0] * 50)      # rule always fires
    y = pd.Series([0] * 50, index=X.index)              # but nothing to learn

    assert MetaLabel(model_zoo(0)["logistic"]).fit(X, y).fitted_ is False


def test_zoo_is_the_five_models_and_every_one_clones():
    zoo = model_zoo(20261014)

    assert list(zoo) == ["buy_and_hold", "ma_crossover", "logistic", "gbt",
                         "meta_crossover_logistic"]
    for name, model in zoo.items():
        assert clone(model) is not model


def test_every_model_returns_probabilities_in_range():
    X, y = meta_fixture()
    for name, model in model_zoo(0).items():
        proba = model.fit(X, y).predict_proba(X)

        assert proba.shape == (len(X), 2)
        assert ((proba >= 0.0) & (proba <= 1.0)).all()
        assert np.allclose(proba.sum(axis=1), 1.0)


def test_no_signal_floor_is_documented_above_a_half():
    # Real-data AUC is read against this, not against 0.50. If it ever drifts
    # back to chance the write-up's framing changes with it.
    from src.synthetic import NO_SIGNAL_FLOOR

    assert 0.50 < NO_SIGNAL_FLOOR < 0.60
