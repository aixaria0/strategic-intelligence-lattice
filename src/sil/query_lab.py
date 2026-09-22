"""Synthetic query-level decision research: crossing payoff curves and safety limits.

A query is one (hypothesized regime, action) Bernoulli simulation. Policy
comparisons consume prefixes of the same cell-wise random streams.
Analytic model probabilities supply independent ground truth for audit.
This model is not a representation of real markets or political actors.
"""
from dataclasses import dataclass
from math import log
import numpy as np

K, A = 3, 3
CELLS = K * A

@dataclass(frozen=True)
class QueryConfig:
    budget: int = 1152
    maximum: int = 256
    pilot: int = 16
    batch: int = 16
    threshold: float = 0.75
    delta: float = 0.05
    seed: int = 2026

    def __post_init__(self):
        if not 8 <= self.pilot <= self.maximum <= 100000:
            raise ValueError("bad pilot or maximum")
        if not 1 <= self.batch <= self.maximum:
            raise ValueError("bad batch")
        if not CELLS * self.pilot <= self.budget <= CELLS * self.maximum:
            raise ValueError("budget outside pilot..full-grid range")
        if not 0 < self.threshold < 1 or not 0 < self.delta < 1:
            raise ValueError("bad threshold or delta")
        if self.seed < 0:
            raise ValueError("negative seed")

@dataclass(frozen=True)
class Scenario:
    identifier: int
    demand: float
    stress: float
    prior: tuple
    posterior: tuple
    evidence: float
    true_regime: int
    reward_prob: np.ndarray
    safety_prob: np.ndarray
    threshold: float

    @property
    def oracle_action(self):
        feasible = [0] + [a for a in (1, 2)
                          if np.all(self.safety_prob[:, a] >= self.threshold)]
        score = np.asarray(self.posterior) @ self.reward_prob
        return int(max(feasible, key=lambda a: float(score[a])))

def make_scenario(seed, identifier, threshold=0.75):
    rng = np.random.default_rng(np.random.SeedSequence([seed, identifier, 4819]))
    demand, stress = float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1))
    prior = rng.dirichlet([1.2, 1.1, 1.3])
    hidden = int(rng.choice(K, p=prior))
    evidence = float(rng.normal(hidden, 0.65))
    likelihood = np.exp(-0.5 * ((evidence - np.arange(K)) / 0.65) ** 2)
    posterior = prior * likelihood
    posterior /= posterior.sum()
    m = np.arange(K, dtype=float)
    reward = np.empty((K, A))
    safe = np.empty((K, A))
    reward[:, 0] = 0.48 + 0.035 * demand - 0.020 * stress + 0.012 * (m - 1)
    reward[:, 1] = 0.565 + 0.145 * demand - 0.020 * stress - 0.070 * (m - 1)
    reward[:, 2] = 0.565 - 0.145 * demand + 0.035 * stress + 0.070 * (m - 1)
    safe[:, 0] = 1.0  # Safe fallback is a specified property of THIS toy.
    safe[:, 1] = np.clip(0.945 - 0.040 * stress - 0.045 * m +
                          0.018 * demand, 0.65, 0.998)
    safe[:, 2] = np.clip(0.90 - 0.085 * stress - 0.075 * m -
                          0.020 * demand, 0.55, 0.997)
    return Scenario(identifier, demand, stress, tuple(prior), tuple(posterior),
                    evidence, hidden, np.clip(reward, 0.03, 0.97), safe, threshold)

def streams(scenario, cfg):
    r = np.empty((K, A, cfg.maximum), dtype=np.uint8)
    s = np.empty_like(r)
    for m in range(K):
        for a in range(A):
            rng = np.random.default_rng(np.random.SeedSequence(
                [cfg.seed, scenario.identifier, m, a, 9183]))
            r[m, a] = rng.random(cfg.maximum) < scenario.reward_prob[m, a]
            s[m, a] = rng.random(cfg.maximum) < scenario.safety_prob[m, a]
    return r, s

def assess(reward_sum, safety_sum, counts, scenario, cfg):
    reward_mean = reward_sum / counts
    safe_mean = safety_sum / counts
    # Uniform Hoeffding union bound over all cells AND all sample sizes.
    # Valid for iid bounded observations and fixed pre-simulation posterior.
    radius = np.sqrt(log(4 * CELLS * cfg.maximum / cfg.delta) / (2.0 * counts))
    value_lower = np.clip(reward_mean - radius, 0.0, 1.0)
    value_upper = np.clip(reward_mean + radius, 0.0, 1.0)
    safe_lower = np.clip(safe_mean - radius, 0.0, 1.0)
    safe_upper = np.clip(safe_mean + radius, 0.0, 1.0)
    safe_lower[:, 0], safe_upper[:, 0] = 1.0, 1.0
    certified = [0] + [a for a in (1, 2)
                       if bool(np.all(safe_lower[:, a] >= cfg.threshold))]
    possible = [0] + [a for a in (1, 2)
                      if bool(np.all(safe_upper[:, a] >= cfg.threshold))]
    w = np.asarray(scenario.posterior)
    scores, lows, highs = w @ reward_mean, w @ value_lower, w @ value_upper
    picked = int(max(certified, key=lambda a: float(scores[a])))
    rivals = [a for a in possible if a != picked]
    done = not rivals or all(float(lows[picked]) > float(highs[a]) for a in rivals)
    return dict(action=picked, certified=certified, possible=possible,
                scores=scores, reward_mean=reward_mean, safe_mean=safe_mean,
                safe_lower=safe_lower, certificate=bool(done))

def priority(counts, status, scenario, cfg):
    scores = status["scores"]
    top = int(np.argmax(scores))
    other = max((a for a in range(A) if a != top),
                key=lambda a: float(scores[a]))
    gap = float(abs(scores[top] - scores[other]))
    rank = 1.0 / (0.025 + np.abs(scores[None, :] - scores[top]))
    rank[:, top] = rank[:, other] = 1.0 / (0.025 + gap)
    weights = np.asarray(scenario.posterior)[:, None]
    mean = status["reward_mean"]
    uncertainty = weights * rank * np.sqrt(
        np.maximum(0.01, mean * (1 - mean)) / counts)
    boundary = 1.0 / ((0.02 + np.abs(
        status["safe_mean"] - cfg.threshold)) * np.sqrt(counts))
    boundary[:, 0] = 0.0
    result = uncertainty + 0.08 * boundary + 0.005 / np.sqrt(counts)
    result[counts >= cfg.maximum] = -np.inf
    return result

def allocate(scenario, cfg, method):
    if method not in ("fixed", "uniform", "random", "adaptive"):
        raise ValueError("unknown method")
    reward_stream, safe_stream = streams(scenario, cfg)
    counts = np.zeros((K, A), dtype=int)
    reward_sum = np.zeros((K, A), dtype=int)
    safe_sum = np.zeros((K, A), dtype=int)
    cap = CELLS * cfg.maximum if method == "fixed" else cfg.budget
    chooser = np.random.default_rng(np.random.SeedSequence(
        [cfg.seed, scenario.identifier, 73919]))
    def sample(m, a, n):
        lo, hi = counts[m, a], counts[m, a] + n
        reward_sum[m, a] += int(reward_stream[m, a, lo:hi].sum())
        safe_sum[m, a] += int(safe_stream[m, a, lo:hi].sum())
        counts[m, a] = hi
    for m in range(K):
        for a in range(A):
            sample(m, a, cfg.pilot)
    used = int(counts.sum())
    stop = "budget_exhausted"
    while used < cap:
        if method in ("fixed", "uniform"):
            candidates = -counts.astype(float)
        elif method == "random":
            candidates = chooser.random((K, A))
        else:
            current = assess(reward_sum, safe_sum, counts, scenario, cfg)
            if current["certificate"]:
                stop = "simultaneous_bound_certificate"
                break
            candidates = priority(counts, current, scenario, cfg)
        candidates[counts >= cfg.maximum] = -np.inf
        idx = int(np.argmax(candidates))
        if not np.isfinite(candidates.flat[idx]):
            stop = "per_cell_limit"
            break
        m, a = np.unravel_index(idx, candidates.shape)
        n = min(cfg.batch, cfg.maximum - int(counts[m, a]), cap - used)
        if n <= 0:
            stop = "per_cell_limit"
            break
        sample(m, a, int(n))
        used += int(n)
    final = assess(reward_sum, safe_sum, counts, scenario, cfg)
    a = final["action"]
    # Research-only exploration may select a statistically UNcertified action.
    # It must NEVER replace the certified selection in a real execution path.
    empirically_feasible = [0] + [a0 for a0 in (1, 2)
        if np.all(final["safe_mean"][:, a0] >= cfg.threshold)]
    exploratory = int(max(empirically_feasible,
                          key=lambda idx: float(final["scores"][idx])))
    value = np.asarray(scenario.posterior) @ scenario.reward_prob
    exploratory_safe = bool(np.all(
        scenario.safety_prob[:, exploratory] >= cfg.threshold))
    return dict(method=method, scenario=scenario.identifier,
                budget=cap, used=used, counts=counts.tolist(),
                action=a, oracle=scenario.oracle_action,
                oracle_value=float(value[scenario.oracle_action]),
                selected_true_value=float(value[a]),
                regret=max(0.0, float(value[scenario.oracle_action] - value[a])),
                truly_safe=bool(np.all(scenario.safety_prob[:, a] >= cfg.threshold)),
                exploratory_action=exploratory,
                exploratory_truly_safe=exploratory_safe,
                exploratory_oracle_agreement=bool(exploratory == scenario.oracle_action),
                exploratory_regret_if_safe=(
                    max(0.0, float(value[scenario.oracle_action] - value[exploratory]))
                    if exploratory_safe else None),
                exploratory_utility=float(value[exploratory]),
                exploration_is_not_authorized=True,
                certificate=bool(final["certificate"]), stop_reason=stop,
                certified_actions=final["certified"],
                possibly_safe_actions=final["possible"],
                posterior=list(scenario.posterior))
