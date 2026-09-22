"""Predeclared, synthetic query-level allocation invariants and failure tests."""
import numpy as np
import pytest
from sil.query_lab import (
    QueryConfig, make_scenario, streams, assess, allocate, kl_confidence_interval, CELLS, K, A
)


def test_scenarios_have_crossing_oracles_and_bayesian_evidence():
    scenarios = [make_scenario(300, i, 0.75) for i in range(64)]
    assert len({x.oracle_action for x in scenarios}) >= 2
    for x in scenarios:
        assert x.reward_prob.shape == (K, A)
        assert x.safety_prob.shape == (K, A)
        assert np.isclose(sum(x.posterior), 1.0)
        assert np.all(x.safety_prob[:, 0] == 1.0)
        assert np.all((x.reward_prob >= 0) & (x.reward_prob <= 1))


def test_budget_and_sample_cap_and_reproducibility():
    cfg = QueryConfig(budget=360, maximum=128, pilot=16, batch=16, seed=75)
    scenario = make_scenario(cfg.seed, 11, cfg.threshold)
    fixed = allocate(scenario, cfg, "fixed")
    assert fixed["used"] == CELLS * cfg.maximum
    for name in ("adaptive", "random", "uniform"):
        result = allocate(scenario, cfg, name)
        assert result == allocate(scenario, cfg, name)
        assert result["used"] <= cfg.budget
        assert result["used"] == sum(map(sum, result["counts"]))
        assert min(min(row) for row in result["counts"]) >= cfg.pilot
        assert max(max(row) for row in result["counts"]) <= cfg.maximum
        assert result["action"] in range(A)
        assert result["truly_safe"]
        assert result["exploration_is_not_authorized"] is True
        assert result["exploratory_action"] in range(A)
        assert result["exploratory_regret_if_safe"] is None or result["exploratory_regret_if_safe"] >= 0
    with pytest.raises(ValueError):
        allocate(scenario, cfg, "unknown")


def test_every_allocator_sees_same_seeded_query_streams():
    cfg = QueryConfig(maximum=64, budget=288)
    scenario = make_scenario(cfg.seed, 5)
    x, y = streams(scenario, cfg)
    p, q = streams(scenario, cfg)
    assert np.array_equal(x, p)
    assert np.array_equal(y, q)


def test_conservative_fallback_and_certificate_not_assumed():
    cfg = QueryConfig(budget=144, maximum=16, pilot=16, batch=16,
                      threshold=0.95, seed=9)
    scenario = make_scenario(cfg.seed, 1, cfg.threshold)
    result = allocate(scenario, cfg, "adaptive")
    assert result["action"] == 0
    assert result["truly_safe"]
    assert result["used"] == 144


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        QueryConfig(budget=100)
    with pytest.raises(ValueError):
        QueryConfig(maximum=8, pilot=16)
    with pytest.raises(ValueError):
        QueryConfig(threshold=1.0)


def test_no_fabricated_true_value_when_high_risk_action_unavailable():
    cfg = QueryConfig(budget=288, maximum=64, threshold=0.9)
    x = make_scenario(cfg.seed, 17, cfg.threshold)
    result = allocate(x, cfg, "fixed")
    selected = result["action"]
    feasible = [0] + [a for a in (1, 2)
                      if np.all(x.safety_prob[:, a] >= cfg.threshold)]
    assert result["oracle"] in feasible
    assert selected in feasible  # simultaneous conservative lower bound or a=0
    assert result["regret"] >= 0


def test_exploratory_signal_is_reported_separately_from_certified_selection():
    cfg = QueryConfig(budget=576, maximum=128, seed=199)
    scenario = make_scenario(cfg.seed, 8, cfg.threshold)
    result = allocate(scenario, cfg, "adaptive")
    assert result["action"] in result["certified_actions"]
    assert result["exploratory_action"] in range(A)
    assert result["exploration_is_not_authorized"]
    if not result["exploratory_truly_safe"]:
        assert result["exploratory_regret_if_safe"] is None


def test_time_uniform_kl_intervals_are_conservative_at_known_extremes():
    cfg = QueryConfig(maximum=256, budget=1152, threshold=0.75)
    counts = np.full((K, A), 256, dtype=int)
    successes = counts.copy()
    lower, upper = kl_confidence_interval(successes, counts, cfg)
    assert np.all(lower <= 1.0)
    assert np.all(upper >= 1.0 - 1e-10)
    assert np.all(lower > 0.90)
    zero, high = kl_confidence_interval(np.zeros_like(counts), counts, cfg)
    assert np.all(zero <= 1e-10)
    assert np.all(high < 0.10)
    assert np.all(np.isfinite(lower))
    assert np.all(np.isfinite(high))
