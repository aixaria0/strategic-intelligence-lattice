"""Budget-aware Monte Carlo allocation across synthetic opponent models.

The selector samples all actions within a chosen model using the SAME shocks,
then chooses which model needs another batch. It is a heuristic decision-focused
racing algorithm: stop_gap is not a formal best-arm identification certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .core import ACTIONS, MODEL_STRENGTHS, SIGMA, Config, discrete_entropy, reward


@dataclass(frozen=True)
class Allocation:
    budget: int
    pilot: int = 16
    batch: int = 64
    stop_gap: float = 2.0

    def validate(self, cfg: Config) -> None:
        minimum = len(ACTIONS) * len(MODEL_STRENGTHS) * self.pilot
        maximum = len(ACTIONS) * len(MODEL_STRENGTHS) * cfg.trials
        if not 16 <= self.pilot <= cfg.trials:
            raise ValueError("pilot must be between 16 and cfg.trials")
        if not 1 <= self.batch <= 100_000:
            raise ValueError("batch must be between 1 and 100000; per-model sampling is capped separately")
        if self.budget < minimum or self.budget > maximum:
            raise ValueError(f"budget must be {minimum}..{maximum} action-rollouts")
        if not np.isfinite(self.stop_gap) or self.stop_gap <= 0:
            raise ValueError("stop_gap must be finite and positive")


def _statistics(terminals: np.ndarray, minimum: np.ndarray,
                action: float) -> dict:
    utilities = reward(terminals, action)
    favored = (terminals[:, 0] >= 0.60) & (terminals[:, 1] >= 0.20)
    safe = minimum >= 0.15
    entropy = (discrete_entropy(terminals[favored])
               if np.count_nonzero(favored) >= 12 else 1.0)
    return {
        "utility": float(utilities.mean()),
        "shortfall": float(max(0.0, 0.65 - np.quantile(utilities, 0.10))),
        "safe": float(np.mean(safe)),
        "favorable_rate": float(np.mean(favored)),
        "favorable_count": int(np.count_nonzero(favored)),
        "favorable_entropy": entropy,
        "utility_standard_deviation": float(utilities.std(ddof=1)) if len(utilities) > 1 else 0.0,
        "n": len(utilities),
    }


def _score_rows(samples: list[list[list[tuple[np.ndarray, np.ndarray]]]],
                weights: np.ndarray, alpha: float, beta: float,
                safety_threshold: float) -> tuple[list[dict], list[list[dict]]]:
    """Rebuild current score with same empirical definitions as fixed evaluate()."""
    model_stats: list[list[dict]] = []
    for model_samples in samples:
        by_action = []
        for action_id, chunks in enumerate(model_samples):
            terminal = np.concatenate([pair[0] for pair in chunks], axis=0)
            minimum = np.concatenate([pair[1] for pair in chunks], axis=0)
            by_action.append(_statistics(terminal, minimum, float(ACTIONS[action_id])))
        model_stats.append(by_action)

    rows: list[dict] = []
    for aid, action in enumerate(ACTIONS):
        ms = [model_stats[m][aid] for m in range(len(MODEL_STRENGTHS))]
        def avg(key: str) -> float:
            return float(np.dot(weights, [m[key] for m in ms]))
        minimum_safe = min(m["safe"] for m in ms)
        expected, downside, entropy = avg("utility"), avg("shortfall"), avg("favorable_entropy")
        rows.append({
            "action": float(action),
            "score": float(expected - alpha * downside - beta * 0.10 * entropy),
            "expected_reward": expected,
            "downside": downside,
            "favorable_rate": avg("favorable_rate"),
            "conditional_entropy": entropy,
            "min_model_safety": minimum_safe,
            "feasible": minimum_safe >= safety_threshold,
            "model_details": ms,
        })
    return rows, model_stats


def _priority(model_stats: list[list[dict]], weights: np.ndarray,
              leader: int, runner: int, minimum_safety: float) -> np.ndarray:
    """Heuristic information-per-simulated-rollout proxy, not exact VOI."""
    scores = []
    for m, action_stats in enumerate(model_stats):
        first, second = action_stats[leader], action_stats[runner]
        n = first["n"]
        # Due to common random numbers a paired gap may be less noisy than this
        # intentionally conservative independent-arm standard-error proxy.
        gap_se = np.hypot(first["utility_standard_deviation"],
                          second["utility_standard_deviation"]) / np.sqrt(n)
        near_constraint = max(
            1.0 / (1.0 + 8.0 * abs(s["safe"] - minimum_safety))
            for s in action_stats
        )
        scores.append(float(weights[m] * (gap_se + 0.25 / np.sqrt(n)) +
                            0.025 * near_constraint / np.sqrt(n)))
    return np.array(scores, dtype=float)


def evaluate_adaptive(agent, cfg: Config, round_id: int,
                      allocation: Allocation) -> tuple[list[dict], dict]:
    """Budget <= fixed all-model rollout budget; reproducible with seed and round.

    Counts only executed state trajectories (horizon steps) as action-rollouts;
    entropy/priority computation and RNG draws are separate CPU overhead.
    """
    allocation.validate(cfg)
    num_models, num_actions = len(MODEL_STRENGTHS), len(ACTIONS)
    samples: list[list[list[tuple[np.ndarray, np.ndarray]]]] = [
        [[] for _ in ACTIONS] for _ in MODEL_STRENGTHS
    ]
    generators = [
        np.random.default_rng(np.random.SeedSequence(
            [cfg.seed, round_id, model_id, 1729]))
        for model_id in range(num_models)
    ]
    model_trials = np.zeros(num_models, dtype=int)
    used = 0

    def sample(model_id: int, count: int) -> None:
        nonlocal used
        shocks = generators[model_id].normal(
            0, SIGMA, size=(count, cfg.horizon, 2))
        # Vectorize ACTIONS too: a single horizon loop for the whole
        # action-by-trial batch, instead of a Python horizon loop per action.
        strength = float(MODEL_STRENGTHS[model_id])
        drift = np.stack((
            0.015 + 0.080 * ACTIONS - 0.052 * strength,
            0.020 - 0.065 * ACTIONS - 0.012 * strength,
        ), axis=-1)[:, None, :]
        states = np.broadcast_to(agent.state, (num_actions, count, 2)).copy()
        minimum = states[:, :, 1].copy()
        for t in range(cfg.horizon):
            states = np.clip(states + drift + shocks[None, :, t, :], 0.0, 1.0)
            minimum = np.minimum(minimum, states[:, :, 1])
        for aid in range(num_actions):
            samples[model_id][aid].append((states[aid].copy(), minimum[aid].copy()))
        model_trials[model_id] += count
        used += num_actions * count

    for model_id in range(num_models):
        sample(model_id, allocation.pilot)
    stop_reason = "budget_exhausted"
    last_gap, last_width = None, None
    while True:
        rows, stats = _score_rows(samples, agent.prior, agent.alpha, agent.beta,
                                  cfg.min_safe_probability)
        feasible = [aid for aid, row in enumerate(rows) if row["feasible"]]
        ordered = sorted(feasible if feasible else list(range(num_actions)),
                         key=lambda aid: rows[aid]["score"], reverse=True)
        leader, runner = ordered[0], ordered[1] if len(ordered) > 1 else ordered[0]
        gap = float(rows[leader]["score"] - rows[runner]["score"])
        weighted_variance = sum(
            (float(agent.prior[m]) ** 2) *
            (stats[m][leader]["utility_standard_deviation"] ** 2 +
             stats[m][runner]["utility_standard_deviation"] ** 2) /
            int(model_trials[m])
            for m in range(num_models)
        )
        mean_width = allocation.stop_gap * np.sqrt(max(0.0, weighted_variance))
        # Explicit unestimated entropy and downside terms: this is still
        # a stopping *heuristic*, not a statistical confidence interval.
        entropy_allowance = 0.20 * max(0.0, agent.beta)
        downside_allowance = (max(0.0, agent.alpha) * 0.69 /
                              np.sqrt(int(min(model_trials))))
        effective_width = float(mean_width + entropy_allowance + downside_allowance)
        last_gap, last_width = gap, effective_width

        stable_safety = all(
            abs(model_row["safe"] - cfg.min_safe_probability) >
            2.0 * np.sqrt((model_row["safe"] * (1-model_row["safe"]) + 0.01) /
                          model_row["n"])
            for model_rows in stats for model_row in model_rows
        )
        if (len(feasible) >= 2 and min(model_trials) >= max(32, allocation.pilot)
                and stable_safety and gap > effective_width):
            stop_reason = "decision_gap_heuristic"
            break

        remaining = allocation.budget - used
        if remaining < num_actions or not np.any(model_trials < cfg.trials):
            break
        priorities = _priority(stats, agent.prior, leader, runner,
                               cfg.min_safe_probability)
        priorities[model_trials >= cfg.trials] = -np.inf
        selected_model = int(np.argmax(priorities))
        count = min(allocation.batch,
                    int(cfg.trials - model_trials[selected_model]),
                    remaining // num_actions)
        if count < 1:
            break
        sample(selected_model, count)

    # rows already describe the final sample collection: no redundant histogram pass.
    return rows, {
        "allocation_mode": "adaptive",
        "rollouts_used": int(used),
        "rollouts_cap": int(num_models * num_actions * cfg.trials),
        "budget_requested": int(allocation.budget),
        "model_trials": model_trials.tolist(),
        "stop_reason": stop_reason,
        "decision_gap": last_gap,
        "decision_uncertainty_proxy": last_width,
        "heuristic_not_certificate": True,
    }
