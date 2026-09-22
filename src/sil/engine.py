"""Fair seeded multi-agent toy tournament, common shocks and separate fixed utility."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .core import ACTIONS, MODEL_STRENGTHS, SIGMA, Config, discrete_entropy, make_shocks, reward, rollout, step
from .inference import posterior

@dataclass
class Agent:
    name: str
    alpha: float
    beta: float
    prior: np.ndarray = field(default_factory=lambda: np.ones(3) / 3)
    state: np.ndarray = field(default_factory=lambda: np.array([0.48, 0.60]))
    cumulative_reward: float = 0.0
    cumulative_baseline: float = 0.0
    cumulative_uplift: float = 0.0
    total_safety_violations: int = 0

    def __post_init__(self) -> None:
        self.prior = np.asarray(self.prior, dtype=float).copy()
        self.state = np.asarray(self.state, dtype=float).copy()

def evaluate(agent: Agent, cfg: Config, round_id: int) -> list[dict]:
    """Empirical safety gate per model, never a formal safety certificate."""
    by_action: list[list[dict]] = [[] for _ in ACTIONS]
    for model_id, strength in enumerate(MODEL_STRENGTHS):
        shocks = make_shocks(cfg, round_id, model_id)
        for aid, action in enumerate(ACTIONS):
            terminals, minimum_reserve = rollout(agent.state, float(action), float(strength), shocks)
            utility = reward(terminals, float(action))
            favored = (terminals[:, 0] >= 0.60) & (terminals[:, 1] >= 0.20)
            safe = minimum_reserve >= 0.15
            entropy = discrete_entropy(terminals[favored]) if np.count_nonzero(favored) >= 12 else 1.0
            by_action[aid].append({
                "utility": float(utility.mean()),
                "shortfall": float(max(0.0, 0.65 - np.quantile(utility, 0.10))),
                "safe": float(np.mean(safe)),
                "favorable_rate": float(np.mean(favored)),
                "favorable_count": int(np.count_nonzero(favored)),
                "favorable_entropy": entropy,
            })
    results = []
    for aid, action in enumerate(ACTIONS):
        ms = by_action[aid]
        weights = agent.prior
        def avg(key: str) -> float:
            return float(np.dot(weights, [m[key] for m in ms]))
        minimum_safe = min(m["safe"] for m in ms)
        expected, shortfall, entropy = avg("utility"), avg("shortfall"), avg("favorable_entropy")
        score = expected - agent.alpha * shortfall - agent.beta * 0.10 * entropy
        results.append({"action": float(action), "score": float(score),
                        "expected_reward": expected, "downside": shortfall,
                        "favorable_rate": avg("favorable_rate"),
                        "conditional_entropy": entropy,
                        "min_model_safety": minimum_safe,
                        "feasible": minimum_safe >= cfg.min_safe_probability,
                        "model_details": ms})
    return results

class Tournament:
    def __init__(self, cfg: Config, num_agents: int = 3, evolve: bool = True):
        if not 2 <= num_agents <= 128:
            raise ValueError("num_agents must be 2..128")
        self.cfg, self.evolve, self.round = cfg, evolve, 0
        self.agents = [Agent(
            name=f"Agent-{i+1}", alpha=0.15 + i * 0.10, beta=0.02 + i * 0.04,
            state=np.array(cfg.initial_state),
            prior=np.array([0.5, 0.3, 0.2]) if i % 2 else np.ones(3)/3,
        ) for i in range(num_agents)]
        self.history: list[dict] = []

    def run_round(self) -> dict:
        self.round += 1
        round_id = self.round
        hidden_model_id = (round_id - 1) % len(MODEL_STRENGTHS)
        true_strength = float(MODEL_STRENGTHS[hidden_model_id])
        realized_shock = np.random.default_rng(np.random.SeedSequence(
            [self.cfg.seed, round_id, 999])).normal(0, SIGMA)
        rows = []
        for agent in self.agents:
            candidates = evaluate(agent, self.cfg, round_id)
            feasible = [x for x in candidates if x["feasible"]]
            chosen = max(feasible, key=lambda x: x["score"]) if feasible else candidates[0]
            action = chosen["action"]
            previous = agent.state.copy()
            observation = step(previous, action, true_strength, realized_shock)
            counterfactual = step(previous, 0.0, true_strength, realized_shock)
            observed_reward = float(reward(observation[None, :], action)[0])
            baseline_reward = float(reward(counterfactual[None, :], 0.0)[0])
            uplift = observed_reward - baseline_reward
            agent.cumulative_reward += observed_reward
            agent.cumulative_baseline += baseline_reward
            agent.cumulative_uplift += uplift
            if observation[1] < 0.15:
                agent.total_safety_violations += 1
            agent.prior = posterior(agent.prior, previous, action, observation)
            agent.state = observation
            rows.append({
                "agent": agent.name, "action": action,
                "realized_reward": observed_reward, "baseline_reward": baseline_reward,
                "realized_uplift": uplift, "cumulative_uplift": agent.cumulative_uplift,
                "cumulative_reward": agent.cumulative_reward,
                "min_model_safety": chosen["min_model_safety"],
                "empirical_feasible": bool(chosen["feasible"]),
                "conditional_entropy": chosen["conditional_entropy"],
                "favorable_rate": chosen["favorable_rate"],
                "model_weights": agent.prior.tolist(),
                "state": observation.tolist(), "alpha": agent.alpha,
                "beta": agent.beta, "safety_violations": agent.total_safety_violations,
            })
        if self.evolve and len(self.agents) > 2 and round_id % 3 == 0:
            ordered = sorted(self.agents, key=lambda a: a.cumulative_reward, reverse=True)
            leader, explorer = ordered[0], ordered[-1]
            rng = np.random.default_rng(np.random.SeedSequence([self.cfg.seed, round_id, 31337]))
            explorer.alpha = float(np.clip(leader.alpha + rng.normal(0, 0.05), 0, 1.5))
            explorer.beta = float(np.clip(leader.beta + rng.normal(0, 0.03), 0, 1.0))
        record = {"round": round_id, "synthetic_true_model": hidden_model_id, "agents": rows}
        self.history.append(record)
        return record

    def run(self, count: int) -> list[dict]:
        if not 1 <= count <= 1_000:
            raise ValueError("round count must be 1..1000")
        return [self.run_round() for _ in range(count)]

    def leaderboard(self) -> list[dict]:
        return [{"name": a.name, "cumulative_reward": round(a.cumulative_reward, 6),
                 "cumulative_uplift": round(a.cumulative_uplift, 6),
                 "safety_violations": a.total_safety_violations,
                 "alpha": round(a.alpha, 4), "beta": round(a.beta, 4)}
                for a in sorted(self.agents, key=lambda x: x.cumulative_reward, reverse=True)]
