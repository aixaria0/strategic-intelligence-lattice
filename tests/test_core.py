import numpy as np
import pytest
from sil.core import Config, discrete_entropy, make_shocks, rollout, MODEL_STRENGTHS
from sil.engine import Tournament, evaluate
from sil.inference import posterior

def test_seed_reproducible_and_common_noise():
    cfg = Config(trials=32, horizon=5, seed=97)
    shocks = make_shocks(cfg, 1, 0)
    assert np.array_equal(shocks, make_shocks(cfg, 1, 0))
    assert not np.array_equal(shocks, make_shocks(cfg, 2, 0))
    x = np.array(cfg.initial_state)
    end1, reserve1 = rollout(x, 0.5, MODEL_STRENGTHS[0], shocks)
    end2, reserve2 = rollout(x, 0.5, MODEL_STRENGTHS[0], shocks)
    assert np.array_equal(end1, end2)
    assert np.array_equal(reserve1, reserve2)
    assert np.all((end1 >= 0) & (end1 <= 1))

def test_empirical_entropy_not_temporal_variance():
    assert discrete_entropy(np.full((64, 2), 0.5)) == pytest.approx(0.0)
    diverse = np.array([[x, y] for x in np.linspace(0.01, 0.99, 5)
                        for y in np.linspace(0.01, 0.99, 5)])
    assert discrete_entropy(diverse) > 0.9
    assert discrete_entropy(np.empty((0, 2))) == 0.0

def test_bayesian_update_is_normalized_and_uses_observation():
    prior = np.ones(3) / 3
    state = np.array([0.48, 0.60])
    from sil.core import step
    observation = step(state, 0.5, MODEL_STRENGTHS[2], np.zeros(2))
    weights = posterior(prior, state, 0.5, observation)
    assert np.isclose(weights.sum(), 1.0)
    assert weights[2] > weights[0]
    assert np.isfinite(weights).all()

def test_evaluation_and_safety_gate():
    t = Tournament(Config(trials=32, horizon=5, min_safe_probability=1.0), 3)
    rows = evaluate(t.agents[0], t.cfg, 1)
    assert len(rows) == 3
    assert all(0 <= row['conditional_entropy'] <= 1 for row in rows)
    assert all(0 <= row['min_model_safety'] <= 1 for row in rows)
    assert all(len(row['model_details']) == 3 for row in rows)

def test_tournament_reproducible_and_metrics_are_realized():
    cfg = Config(trials=32, horizon=5, seed=9)
    a, b = Tournament(cfg), Tournament(cfg)
    a.run(4)
    b.run(4)
    assert a.history == b.history
    assert a.leaderboard() == b.leaderboard()
    assert np.allclose([x.prior.sum() for x in a.agents], 1.0)
    for agent in a.agents:
        assert agent.cumulative_uplift == pytest.approx(
            agent.cumulative_reward - agent.cumulative_baseline)

def test_bad_config():
    with pytest.raises(ValueError):
        Config(trials=0)
    with pytest.raises(ValueError):
        Tournament(Config(), num_agents=1)

def test_policy_weights_do_not_change_objective_or_common_random_outcomes():
    cfg = Config(trials=32, horizon=4, seed=40)
    first = Tournament(cfg, num_agents=2, evolve=False)
    other = Tournament(cfg, num_agents=2, evolve=False)
    first.agents[0].alpha, first.agents[0].beta = 0.0, 0.0
    other.agents[0].alpha, other.agents[0].beta = 1.5, 1.0
    a, b = evaluate(first.agents[0], cfg, 1), evaluate(other.agents[0], cfg, 1)
    assert [x['expected_reward'] for x in a] == [x['expected_reward'] for x in b]
    assert [x['score'] for x in a] != [x['score'] for x in b]

def test_explicit_fallback_when_every_policy_fails_safety():
    cfg = Config(trials=32, horizon=12, seed=45,
                 initial_state=(0.48, 0.16), min_safe_probability=1.0)
    lab = Tournament(cfg, num_agents=2, evolve=False)
    record = lab.run_round()
    assert all(row['action'] == 0.0 for row in record['agents'])
    assert all(row['empirical_feasible'] is False for row in record['agents'])
