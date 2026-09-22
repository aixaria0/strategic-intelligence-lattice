"""Exact posterior-predictive reward EVSI under the declared Beta-Bernoulli model."""
import numpy as np
import pytest

from sil.evsi import beta_binomial_pmf, reward_batch_evsi
from sil.query_lab import QueryConfig, allocate, make_scenario


def test_beta_binomial_uniform_prior_and_normalization():
    assert np.allclose(beta_binomial_pmf(1.0, 1.0, 2),
                       np.ones(3) / 3)
    for alpha, beta, n in [(2, 3, 16), (120, 8, 16), (1, 1, 128)]:
        p = beta_binomial_pmf(alpha, beta, n)
        assert np.isclose(p.sum(), 1.0)
        assert np.all(p >= 0)
    with pytest.raises(ValueError):
        beta_binomial_pmf(0, 1, 2)


def test_one_query_evsi_has_exact_known_value():
    # Two initially identical Beta(1,1) action rewards. A one-trial query
    # improves posterior optimal reward by 1/12 exactly, under this model.
    counts = np.zeros((1, 2), dtype=int)
    sums = np.zeros_like(counts)
    info = reward_batch_evsi(sums, counts, np.array([1.0]), 1)
    assert np.allclose(info, np.full((1, 2), 1.0 / 12.0), atol=1e-12)
    assert np.all(info >= 0)


def test_dominant_action_has_zero_one_step_reward_information():
    counts = np.full((1, 2), 100, dtype=int)
    successes = np.array([[100, 0]])
    info = reward_batch_evsi(successes, counts, np.array([1.0]), 1)
    assert np.allclose(info, 0.0, atol=1e-12)


def test_evsi_mode_obeys_shared_seed_budget_and_safety_gate():
    cfg = QueryConfig(budget=288, maximum=64, seed=178)
    scenario = make_scenario(cfg.seed, 5, cfg.threshold)
    a = allocate(scenario, cfg, "evsi_reward")
    b = allocate(scenario, cfg, "evsi_reward")
    assert a == b
    assert a["used"] <= cfg.budget
    assert a["truly_safe"]
    assert a["exploration_is_not_authorized"]
    assert a["method"] == "evsi_reward"


def test_evsi_prices_last_partial_batch_without_overspending():
    cfg = QueryConfig(maximum=64, pilot=16, batch=16, budget=145, seed=311)
    scenario = make_scenario(cfg.seed, 1, cfg.threshold)
    result = allocate(scenario, cfg, "evsi_reward")
    assert result["used"] == 145
    assert sum(map(sum, result["counts"])) == 145
    assert result["truly_safe"]
