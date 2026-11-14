import numpy as np
import pandas as pd
import pytest
from scipy.stats import kurtosis, norm, skew

from src.metrics import (
    deflated_sharpe, expected_max_sharpe, max_drawdown, sharpe, sharpe_se,
    summary,
)


def test_sharpe_against_a_hand_computed_value():
    r = pd.Series([0.01, -0.01, 0.02, 0.00, 0.01])
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)

    assert sharpe(r) == pytest.approx(expected)


def test_sharpe_annualises_by_root_periods():
    # The square root is there because variances add over days, not standard
    # deviations. Multiplying by 252 instead is the classic version of this bug.
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 2000))

    assert sharpe(r, periods=252) == pytest.approx(sharpe(r, periods=1) * np.sqrt(252))


def test_flat_returns_give_zero_not_a_division_error():
    r = pd.Series([0.0] * 50)

    assert sharpe(r) == 0.0
    assert sharpe_se(r) == 0.0


def test_sharpe_se_matches_lo_2002():
    r = pd.Series([0.01, -0.01, 0.02, 0.00, 0.01] * 40)
    sr = r.mean() / r.std(ddof=1)
    expected = np.sqrt((1 + 0.5 * sr ** 2) / len(r)) * np.sqrt(252)

    assert sharpe_se(r) == pytest.approx(expected)


def test_sharpe_se_shrinks_with_more_data():
    # The reason our error bars are wide is the sample, not the strategy.
    rng = np.random.default_rng(1)
    short = pd.Series(rng.normal(0.0005, 0.01, 250))
    long_ = pd.Series(rng.normal(0.0005, 0.01, 4000))

    assert sharpe_se(short) > sharpe_se(long_)


def test_max_drawdown_on_a_known_path():
    # 1.0 -> 1.2 -> 0.6: the worst peak-to-trough is 50%.
    r = pd.Series([0.2, -0.5, 0.1])

    assert max_drawdown(r) == pytest.approx(0.5)


def test_max_drawdown_of_a_rising_line_is_zero():
    assert max_drawdown(pd.Series([0.01] * 50)) == pytest.approx(0.0)


def test_expected_max_sharpe_grows_with_trials_and_spread():
    # Both arguments matter: twenty near-identical configurations buy far less
    # luck than twenty wildly different ones.
    assert expected_max_sharpe(50, 0.05) > expected_max_sharpe(5, 0.05)
    assert expected_max_sharpe(5, 0.10) > expected_max_sharpe(5, 0.05)


def test_expected_max_sharpe_is_zero_without_a_search():
    assert expected_max_sharpe(1, 0.05) == 0.0
    assert expected_max_sharpe(5, 0.0) == 0.0


def test_expected_max_sharpe_matches_the_published_formula():
    n, sd = 20, 0.04
    g = 0.5772156649015329
    expected = sd * ((1 - g) * norm.ppf(1 - 1 / n)
                     + g * norm.ppf(1 - 1 / (n * np.e)))

    assert expected_max_sharpe(n, sd) == pytest.approx(expected, rel=1e-6)


def test_deflated_sharpe_is_a_probability():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.0008, 0.01, 3000))

    assert 0.0 <= deflated_sharpe(r, 5, 0.03) <= 1.0


def test_more_trials_deflate_further():
    # The same track record is less impressive if you tried a hundred things.
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0010, 0.01, 3000))

    assert deflated_sharpe(r, 5, 0.03) > deflated_sharpe(r, 100, 0.03)


def test_a_wider_trial_spread_deflates_further():
    rng = np.random.default_rng(4)
    r = pd.Series(rng.normal(0.0010, 0.01, 3000))

    assert deflated_sharpe(r, 20, 0.02) > deflated_sharpe(r, 20, 0.08)


def test_deflation_uses_per_period_returns_not_annualised_ones():
    # Feeding an annualised ratio into the formula inflates it by sqrt(252).
    # Scaling every return by a constant leaves the Sharpe unchanged, so the
    # DSR must not move either.
    rng = np.random.default_rng(5)
    r = pd.Series(rng.normal(0.0008, 0.01, 2000))

    assert deflated_sharpe(r * 10, 5, 0.03) == pytest.approx(
        deflated_sharpe(r, 5, 0.03))


def test_negative_skew_and_fat_tails_lower_the_deflated_ratio():
    # Why the higher moments are in the formula at all: a fat-tailed series
    # lets the luckiest of N look better, so the same Sharpe is worth less.
    # A modest edge, so the deflated ratio is not saturated at 1.0 where
    # nothing can be compared.
    rng = np.random.default_rng(6)
    normal = pd.Series(rng.normal(0.00032, 0.01, 4000))

    # Student-t with 3 degrees of freedom, negated for negative skew and
    # rescaled to the same mean and standard deviation, so the Sharpe is
    # unchanged and only the shape differs.
    t = -rng.standard_t(3, 4000)
    fat = pd.Series((t - t.mean()) / t.std(ddof=1) * normal.std(ddof=1)
                    + normal.mean())

    assert sharpe(fat) == pytest.approx(sharpe(normal), rel=0.02)
    assert skew(fat) < -0.5 and kurtosis(fat, fisher=False) > 5
    assert 0.01 < deflated_sharpe(normal, 5, 0.03) < 0.99
    assert deflated_sharpe(fat, 5, 0.03) < deflated_sharpe(normal, 5, 0.03)


def test_summary_reports_everything_step_8_promises():
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0.0005, 0.01, 500))
    traded = pd.Series(rng.integers(0, 2, 500).astype(float))
    pos = pd.Series(rng.integers(0, 2, 500).astype(float))

    s = summary(r, traded=traded, position=pos, n_trials=5, sr_std=0.03)

    for key in ("sharpe", "sharpe_se", "sharpe_t", "max_drawdown",
                "total_return", "turnover", "exposure", "dsr"):
        assert key in s
    assert s["sharpe_t"] == pytest.approx(s["sharpe"] / s["sharpe_se"])


def test_summary_omits_the_dsr_when_the_search_is_not_described():
    # The DSR is a property of the whole search, not of one backtest, so it
    # cannot be produced from returns alone.
    r = pd.Series(np.random.default_rng(8).normal(0.0005, 0.01, 300))

    assert "dsr" not in summary(r)


def test_exposure_matched_sharpe_agrees_with_itself_at_full_exposure():
    # Nothing to shuffle when every bar is held, so the matched Sharpe must be
    # the real one. This is the check that the benchmark is wired up right.
    from src.metrics import exposure_matched_sharpe

    rng = np.random.default_rng(0)
    r = rng.normal(0.0004, 0.01, 2000)
    pos = np.ones(2000)

    mean, sd = exposure_matched_sharpe(pos, r, n_draws=20)
    assert mean == pytest.approx(sharpe(r), abs=1e-9)
    assert sd == pytest.approx(0.0, abs=1e-9)


def test_exposure_matched_sharpe_isolates_selection_from_exposure():
    # A strategy holding only the best 20% of bars must beat its own
    # exposure-matched benchmark; one holding only the worst 20% must lose to
    # it. Same exposure in both cases, so only selection differs.
    from src.metrics import exposure_matched_sharpe

    rng = np.random.default_rng(1)
    r = rng.normal(0.0004, 0.01, 2000)
    cut = np.quantile(r, 0.8)

    good = (r >= cut).astype(float)
    bad = (r <= np.quantile(r, 0.2)).astype(float)

    good_matched, _ = exposure_matched_sharpe(good, r, n_draws=40)
    bad_matched, _ = exposure_matched_sharpe(bad, r, n_draws=40)

    assert sharpe(np.where(good > 0, r, 0.0)) > good_matched
    assert sharpe(np.where(bad > 0, r, 0.0)) < bad_matched
    # Same number of bars held, so the two benchmarks must be close.
    assert good_matched == pytest.approx(bad_matched, abs=0.15)


def test_exposure_matched_sharpe_checks_its_inputs():
    from src.metrics import exposure_matched_sharpe

    with pytest.raises(ValueError, match="same length"):
        exposure_matched_sharpe(np.ones(10), np.zeros(5))
