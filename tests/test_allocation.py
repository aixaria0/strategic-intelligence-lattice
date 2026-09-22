"""Adaptive Computation Allocation: reproducibility, budget enforcement and parity."""
import numpy as np
import pytest

from sil.allocation import Allocation, evaluate_adaptive
from sil.core import Config, ACTIONS, MODEL_STRENGTHS
from sil.engine import Tournament, evaluate


def test_budget_validation_and_no_overspend():
    cfg = Config(trials=64, horizon=5, seed=7)
    agent = Tournament(cfg, num_agents=2).agents[0]
    with pytest.raises(ValueError):
        evaluate_adaptive(agent, cfg, 1, Allocation(budget=1))
    with pytest.raises(ValueError):
        evaluate_adaptive(agent, cfg, 1, Allocation(budget=10_000))
    rows, report = evaluate_adaptive(agent, cfg, 1, Allocation(budget=256))
    assert len(rows) == len(ACTIONS)
    assert report["rollouts_used"] <= 256
    assert report["rollouts_used"] == len(ACTIONS) * sum(report["model_trials"])
    assert all(16 <= n <= cfg.trials for n in report["model_trials"])
    assert len(report["model_trials"]) == len(MODEL_STRENGTHS)
    assert report["stop_reason"] in ("decision_gap_heuristic", "budget_exhausted")


def test_adaptive_is_deterministic_and_observes_actual_shocks():
    cfg = Config(trials=128, horizon=8, seed=991)
    allocation = Allocation(budget=9 * cfg.trials)
    a, b = Tournament(cfg, num_agents=2), Tournament(cfg, num_agents=2)
    ar, ad = evaluate_adaptive(a.agents[0], cfg, 1, allocation)
    br, bd = evaluate_adaptive(b.agents[0], cfg, 1, allocation)
    assert ar == br
    assert ad == bd
    assert np.isclose(a.agents[0].prior.sum(), 1.0)


def test_full_sampling_matches_fixed_scoring_when_early_stop_disabled():
    cfg = Config(trials=32, horizon=6, seed=183)
    agent = Tournament(cfg, num_agents=2).agents[0]
    expected = evaluate(agent, cfg, 3)
    actual, report = evaluate_adaptive(
        agent, cfg, 3, Allocation(budget=9 * cfg.trials, stop_gap=1.0e8))
    assert report["rollouts_used"] == 9 * cfg.trials
    assert report["model_trials"] == [cfg.trials] * 3
    for fixed, adaptive in zip(expected, actual):
        for field in ("action", "score", "expected_reward", "downside",
                      "conditional_entropy", "favorable_rate", "min_model_safety"):
            assert adaptive[field] == pytest.approx(fixed[field], abs=1e-12)
        assert adaptive["feasible"] == fixed["feasible"]


def test_tournament_adaptive_is_reproducible_and_records_actual_budget():
    cfg = Config(trials=96, horizon=6, seed=35)
    cap = 9 * cfg.trials
    a = Tournament(cfg, num_agents=3, allocation_mode="adaptive",
                   simulation_budget=cap)
    b = Tournament(cfg, num_agents=3, allocation_mode="adaptive",
                   simulation_budget=cap)
    assert a.run(2) == b.run(2)
    assert all(record["total_action_rollouts"] <= 3 * cap for record in a.history)
    assert all(row["simulation"]["allocation_mode"] == "adaptive"
               for record in a.history for row in record["agents"])
    assert all(row["simulation"]["rollouts_used"] <= cap
               for record in a.history for row in record["agents"])
    with pytest.raises(ValueError):
        Tournament(cfg, allocation_mode="broken")
    with pytest.raises(ValueError):
        Tournament(cfg, allocation_mode="fixed", simulation_budget=cap)


def test_nontrivial_gap_may_stop_early_without_claiming_a_certificate():
    cfg = Config(trials=256, horizon=8, seed=163,
                 initial_state=(0.20, 0.80))
    lab = Tournament(cfg, num_agents=2)
    lab.agents[0].alpha = 0.0
    lab.agents[0].beta = 0.0
    rows, report = evaluate_adaptive(
        lab.agents[0], cfg, 1, Allocation(budget=9 * cfg.trials))
    assert report["rollouts_used"] <= 9 * cfg.trials
    assert report["heuristic_not_certificate"] is True
    assert all(np.isfinite(row["score"]) for row in rows)
