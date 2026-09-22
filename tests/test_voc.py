"""C-VoC: safety-aware value of a query and explicit CPU-cost governor.

All guarantees below are CONDITIONAL on synthetic independent Beta-Bernoulli
likelihoods and the separate frequentist safety gate. The one-batch foresight
is not claimed to solve multibatch or real-world value-of-information planning.
"""
import numpy as np
import pytest

from sil.evsi import reward_batch_evsi
from sil.voc import constrained_batch_voc, certified_cells
from sil.query_lab import QueryConfig, make_scenario, allocate


def test_safety_information_can_change_feasible_decision_when_reward_evsi_zero():
    n = np.full((3, 3), 128, dtype=int)
    n[0, 1] = 32
    reward = np.full((3, 3), 64, dtype=int)
    reward[:, 1] = 115
    reward[0, 1] = 29
    safe = np.zeros((3, 3), dtype=int)
    safe[:, 0] = 128
    safe[:, 1] = 128
    safe[0, 1] = 32
    weights = np.ones(3) / 3
    certified = certified_cells(safe, n, .75, .05, 128)
    assert not certified[0, 1]
    assert np.all(certified[1:, 1])
    # A1 is reward-dominant already, so one additional reward-only batch
    # cannot change which action has the highest posterior reward.
    reward_info = reward_batch_evsi(reward, n, weights, 16)
    assert reward_info[0, 1] == pytest.approx(0.0, abs=1e-12)
    future = constrained_batch_voc(reward, safe, n, weights,
                                   .75, .05, 128, 16)
    assert future["current_certified_action"] == 0
    assert 0 < future["future_action_certification_probability"][0, 1] < 1
    assert future["decision_evsi"][0, 1] > 0
    assert future["not_a_safety_certificate"]


def test_forecast_of_known_safe_fallback_does_not_invent_certification():
    n = np.full((3, 3), 16, dtype=int)
    r = np.full((3, 3), 8, dtype=int)
    s = np.zeros_like(n)
    s[:, 0] = 16
    v = constrained_batch_voc(r, s, n, np.ones(3)/3, .9, .05, 128, 8)
    assert v["current_certified_action"] == 0
    assert np.all(np.isfinite(v["decision_evsi"]))
    assert np.all(v["future_action_certification_probability"][:, 0] == 1.0)
    assert np.all(v["future_action_certification_probability"][:, 1:] == 0.0)
    assert np.all(v["expected_positive_improvement"] >= 0)


def test_voc_budget_and_identical_streams_repeatability_without_clock_fields():
    cfg = QueryConfig(budget=320, maximum=64, seed=779)
    scenario = make_scenario(cfg.seed, 4, cfg.threshold)
    a = allocate(scenario, cfg, "c_voc")
    b = allocate(scenario, cfg, "c_voc")
    variable_timing = ("planning_seconds", "sampling_seconds")
    for key in variable_timing:
        a.pop(key)
        b.pop(key)
    assert a == b
    assert a["used"] == cfg.budget
    assert a["lookahead_evaluations"] > 0
    assert a["truly_safe"]
    assert a["action"] in a["certified_actions"]


def test_governor_can_stop_early_only_when_explicitly_enabled():
    cfg = QueryConfig(budget=1152, maximum=128, seed=172,
                      compute_price=1.0)
    scenario = make_scenario(cfg.seed, 3, cfg.threshold)
    result = allocate(scenario, cfg, "c_voc_governor")
    assert result["used"] < cfg.budget
    assert result["stop_reason"] == "governor_one_step_value_below_price"
    assert min(map(min, result["counts"])) >= 64
    assert result["compute_price"] == pytest.approx(1.0)
    assert result["truly_safe"]
    with pytest.raises(ValueError):
        allocate(scenario, QueryConfig(budget=1152, maximum=128),
                 "c_voc_governor")
    with pytest.raises(ValueError):
        QueryConfig(compute_price=-1)
