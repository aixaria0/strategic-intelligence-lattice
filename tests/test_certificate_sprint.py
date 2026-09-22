"""v0.7: cheap multi-model sprint is a query allocator, NOT a risk certificate."""
from dataclasses import replace
import numpy as np
import pytest
from sil.certificate_sprint import choose_certificate_sprint, sprint_schedule
from sil.query_lab import QueryConfig, make_scenario, allocate, streams


def test_sprint_cross_model_target_and_full_budget():
    n = np.full((3, 3), 16, dtype=int)
    rewards = np.full((3, 3), 8, dtype=int)
    rewards[:, 1] = 14
    safety = np.full((3, 3), 16, dtype=int)
    pilot = choose_certificate_sprint(rewards, safety, n, np.ones(3)/3, .75)
    assert pilot is not None
    assert pilot["target_action"] == 1
    assert pilot["not_a_safety_certificate"]
    schedule = sprint_schedule(n, 1, 576, 128)
    projected = n.copy()
    for m, a, amount in schedule:
        projected[m, a] += amount
    assert int(projected.sum()) == 576
    assert all(projected[m, 1] == 128 for m in range(3))
    assert np.all(projected <= 128)
    assert np.all(projected >= 16)


def test_uniform_fallback_batched_without_query_budget_leak():
    n = np.full((3, 3), 16, dtype=int)
    schedule = sprint_schedule(n, None, 577, 128)
    projected = n.copy()
    for m, a, amount in schedule:
        projected[m, a] += amount
    assert int(projected.sum()) == 577
    assert int(projected.max() - projected.min()) <= 1
    with pytest.raises(ValueError):
        sprint_schedule(n, None, 100, 128)


def test_sprint_exact_seeded_draw_prefix_and_final_safety():
    cfg = QueryConfig(seed=661, budget=576, maximum=128)
    scenario = make_scenario(cfg.seed, 5, cfg.threshold)
    r, s = streams(scenario, cfg)
    first = allocate(scenario, cfg, "certificate_sprint")
    second = allocate(scenario, cfg, "certificate_sprint")
    for key in ("planning_seconds", "sampling_seconds"):
        first.pop(key)
        second.pop(key)
    assert first == second
    assert first["used"] == cfg.budget
    assert first["actual_bernoulli_draws"] == 2 * cfg.budget
    assert first["truly_safe"]
    assert first["action"] in first["certified_actions"]
    for m in range(3):
        for a in range(3):
            n = first["counts"][m][a]
            assert first["sampled_reward_successes"][m][a] == int(r[m,a,:n].sum())
            assert first["sampled_safety_successes"][m][a] == int(s[m,a,:n].sum())


def test_low_budget_sprint_no_fabricated_safety():
    cfg = QueryConfig(seed=677, budget=145, maximum=128)
    scenario = make_scenario(cfg.seed, 8, cfg.threshold)
    output = allocate(scenario, cfg, "certificate_sprint")
    assert output["used"] == 145
    assert output["action"] in output["certified_actions"]
    assert output["truly_safe"]
    assert output["exploration_is_not_authorized"]


def test_bulk_uniform_is_identical_to_small_batch_uniform_not_just_cheaper():
    for budget in (288, 576, 577):
        cfg = QueryConfig(seed=913, budget=budget, maximum=128)
        case = make_scenario(cfg.seed, 17, cfg.threshold)
        bulk = allocate(case, cfg, "uniform_bulk")
        small = allocate(case, cfg, "uniform")
        # The nondivisible 577 budget may leave different last-cell locations.
        # At divisible 9-cell budgets, counts and every shared RNG prefix
        # MUST be identical; one algorithm cannot get easier random outcomes.
        if budget % 9 == 0:
            for key in ("counts", "sampled_reward_successes",
                        "sampled_safety_successes", "action", "truly_safe",
                        "regret", "oracle"):
                assert bulk[key] == small[key]
        assert bulk["used"] == small["used"] == budget
        assert bulk["action"] in bulk["certified_actions"]
