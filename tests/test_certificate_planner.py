"""Falsification tests for finite multi-model/multi-batch certification plans."""
import numpy as np
import pytest
from sil.certificate_planner import certification_forecast, plan_certificate_portfolio
from sil.voc import constrained_batch_voc
from sil.query_lab import QueryConfig, make_scenario, allocate


def test_joint_multibatch_value_exists_when_all_single_batch_values_are_zero():
    n = np.full((3, 3), 16, dtype=int)
    reward = np.full((3, 3), 8, dtype=int)
    reward[:, 1] = 14
    safe = np.zeros((3, 3), dtype=int)
    safe[:, 0] = 16
    safe[:, 1] = 16
    weights = np.ones(3) / 3
    # Even perfect results from ONE 16-draw batch cannot pass the
    # time-uniform safety requirement at this starting sample count.
    single = constrained_batch_voc(reward, safe, n, weights, .75, .05, 128, 16)
    assert np.allclose(single["decision_evsi"][:, 1], 0, atol=1e-12)
    plan = plan_certificate_portfolio(
        reward, safe, n, weights, .75, .05, 128, 16, 288)
    assert plan is not None
    assert plan["target_action"] == 1
    assert {q["model"] for q in plan["queries"]} == {0, 1, 2}
    assert all(q["planned_draws"] > 16 for q in plan["queries"])
    assert 0 < plan["joint_certificate_forecast"] <= 1
    assert plan["total_planned_draws"] <= 288
    assert plan["estimated_positive_value"] > 0
    assert plan["not_a_safety_certificate"]


def test_forecast_is_probability_and_depends_on_multibatch_evidence():
    single = certification_forecast(16, 16, 16, .75, .05, 128)
    future = certification_forecast(16, 16, 64, .75, .05, 128)
    assert single == 0
    assert 0 < future <= 1
    assert certification_forecast(0, 16, 0, .75, .05, 128) == 0
    with pytest.raises(ValueError):
        certification_forecast(17, 16, 16, .75, .05, 128)


def test_portfolio_mode_respects_budget_and_final_safety_gate():
    cfg = QueryConfig(budget=320, maximum=64, seed=771)
    scenario = make_scenario(cfg.seed, 4, cfg.threshold)
    a = allocate(scenario, cfg, "certificate_portfolio")
    b = allocate(scenario, cfg, "certificate_portfolio")
    # Wall-clock measurement cannot be deterministically equal.
    for name in ("planning_seconds", "sampling_seconds"):
        a.pop(name)
        b.pop(name)
    assert a == b
    assert a["used"] <= cfg.budget
    assert a["used"] == sum(map(sum, a["counts"]))
    assert all(16 <= x <= cfg.maximum for row in a["counts"] for x in row)
    assert a["action"] in a["certified_actions"]
    assert a["truly_safe"]
    assert a["not_an_optimality_certificate"]


def test_portfolio_never_calls_unsafe_exploratory_option_certified():
    cfg = QueryConfig(budget=576, maximum=128, seed=181)
    scenario = make_scenario(cfg.seed, 11, cfg.threshold)
    output = allocate(scenario, cfg, "certificate_portfolio")
    assert output["truly_safe"]
    assert output["action"] in output["certified_actions"]
    assert output["exploration_is_not_authorized"]
    for plan in output["portfolios"]:
        assert plan["total_planned_draws"] >= 1
        assert plan["estimated_gain_per_query"] > 0
        assert plan["not_a_safety_certificate"]


def test_infeasible_portfolio_does_not_claim_global_zero_information():
    n = np.full((3, 3), 16, dtype=int)
    reward = np.full((3, 3), 8, dtype=int)
    reward[:, 1] = 14
    safe = np.zeros((3, 3), dtype=int)
    safe[:, 0] = 16
    safe[:, 1] = 16
    # All available 16 new simulations cannot certify 3 missing model cells.
    plan = plan_certificate_portfolio(
        reward, safe, n, np.ones(3) / 3, .75, .05, 128, 16, 16)
    assert plan is None
