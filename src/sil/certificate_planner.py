"""Finite multi-query SAFETY CERTIFICATE portfolio planning, experimental v0.6.

A portfolio commits multiple distinct model/action queries whose combined
evidence may make one currently uncertified action admissible. The joint
success forecast uses independent Beta-Binomial safety posteriors.

NOT a globally optimal Bayesian metareasoning solution: reward posterior means
stay fixed during planning, model forecast can be wrong, and candidate lengths
use a coarse grid. Final actions use the existing frequentist KL safety gate.
"""
from __future__ import annotations
from functools import lru_cache
from itertools import product
from math import log
import numpy as np
from .evsi import beta_binomial_pmf
from .voc import bernoulli_kl


@lru_cache(maxsize=32768)
def certification_forecast(successes: int, n: int, extra: int,
                           threshold: float, delta: float, maximum: int,
                           cells: int = 9) -> float:
    """Predict probability of crossing a specified KL safety threshold.

    Exact within the stated independent Beta(1,1) posterior predictive
    Bernoulli cell model, up to floating-point error.
    """
    if not (0 <= successes <= n and n >= 1 and extra >= 0 and
            n + extra <= maximum and 0 < threshold < 1 and 0 < delta < 1):
        raise ValueError("invalid safety forecast inputs")
    future_n = n + extra
    k = np.arange(extra + 1, dtype=float)
    phat = (successes + k) / future_n
    cutoff = log(4 * cells * maximum / delta)
    certified = ((phat > threshold) &
                 (future_n * bernoulli_kl(phat, threshold) > cutoff))
    # Exact early exits: no Bernoulli success count could cross the boundary,
    # or ALL possible counts already cross it. Avoid predictive PMF creation.
    if not bool(np.any(certified)):
        return 0.0
    if bool(np.all(certified)):
        return 1.0
    probs = beta_binomial_pmf(1 + successes, 1 + n - successes, extra)
    return float(np.sum(probs[certified]))


def _sample_options(available: int, batch: int) -> list[int]:
    if available == 0:
        return [0]
    values = {0, available}
    step = batch
    while step < available:
        values.add(step)
        step *= 2
    return sorted(values)


def plan_certificate_portfolio(reward_sum: np.ndarray,
                               safety_sum: np.ndarray,
                               counts: np.ndarray,
                               weights: np.ndarray,
                               threshold: float, delta: float,
                               maximum: int, batch: int, remaining: int,
                               minimum_joint_probability: float = 0.015
                               ) -> dict | None:
    """Rank affordable three-model query portfolios by expected gain per draw.

    A group of uncertain safety cells may be jointly certifiable after
    multiple batches even when a single new batch has zero immediate value.
    Returns None if no plausibly useful affordable plan exists; that is NOT
    evidence that future information has zero value.
    """
    r = np.asarray(reward_sum, dtype=int)
    s = np.asarray(safety_sum, dtype=int)
    n = np.asarray(counts, dtype=int)
    w = np.asarray(weights, dtype=float)
    if (r.shape != (3, 3) or s.shape != r.shape or n.shape != r.shape
            or w.shape != (3,) or not np.isclose(w.sum(), 1)
            or np.any(r < 0) or np.any(s < 0) or np.any(r > n)
            or np.any(s > n) or np.any(n < 1)
            or not 0 < threshold < 1 or not 0 < delta < 1
            or not 1 <= batch <= maximum or remaining < 0):
        raise ValueError("invalid portfolio planner inputs")
    reward_values = w @ ((1.0 + r) / (2.0 + n))
    currently_certified = [0]
    for a in (1, 2):
        if all(certification_forecast(int(s[m, a]), int(n[m, a]), 0,
                                      threshold, delta, maximum) >= 1.0
               for m in range(3)):
            currently_certified.append(a)
    incumbent = max(currently_certified, key=lambda a: float(reward_values[a]))
    incumbent_value = float(reward_values[incumbent])
    best: dict | None = None
    for action in (1, 2):
        advantage = max(0.0, float(reward_values[action]) - incumbent_value)
        if advantage <= 1e-12:
            continue
        choices = []
        for model in range(3):
            available = min(int(maximum - n[model, action]), remaining)
            choices.append([(length, certification_forecast(
                int(s[model, action]), int(n[model, action]), length,
                threshold, delta, maximum))
                for length in _sample_options(available, batch)])
        for combo in product(*choices):
            lengths = [part[0] for part in combo]
            total = sum(lengths)
            if not 0 < total <= remaining:
                continue
            joint = float(np.prod([p for _, p in combo]))
            if joint < minimum_joint_probability:
                continue
            gain = advantage * joint
            rate = gain / total
            if best is None or rate > best["estimated_gain_per_query"]:
                best = {
                    "target_action": int(action),
                    "queries": [{"model": int(m), "action": int(action),
                                 "planned_draws": int(lengths[m])}
                                for m in range(3) if lengths[m] > 0],
                    "total_planned_draws": int(total),
                    "joint_certificate_forecast": joint,
                    "current_reward_advantage": advantage,
                    "estimated_positive_value": gain,
                    "estimated_gain_per_query": float(rate),
                    "planning_assumptions": "independent Beta-Bernoulli safety cells; fixed reward posterior means",
                    "not_a_safety_certificate": True,
                }
    return best
