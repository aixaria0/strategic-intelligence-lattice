"""Constrained Value of Computation (C-VoC), one-query, one-batch lookahead.

A query yields BOTH an independent Bernoulli reward and safety observation.
The posterior predictive distribution is Beta-Binomial in each channel with
independent Beta(1,1) priors. Enumeration of their success counts is exact
UNDER this model. The final risk gate remains the separately defined
time-uniform KL confidence rule; no Bayesian posterior is called a safety
certificate.

The decision-aware query value is the expected change in the best posterior
reward AMONG safety-certified actions. It is not globally optimal sequential
VOI: multiple batches may be needed before any certification can change.
"""
from __future__ import annotations

from math import log
import numpy as np

from .evsi import beta_binomial_pmf


def bernoulli_kl(p: np.ndarray, q: float) -> np.ndarray:
    """D_Bern(p || q), supporting p exactly 0 or 1."""
    x = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (np.where(x > 0, x * np.log(x / q), 0.0) +
                np.where(x < 1, (1 - x) * np.log((1 - x) / (1 - q)), 0.0))


def certified_cells(safety_sum: np.ndarray, counts: np.ndarray,
                    threshold: float, delta: float,
                    maximum: int) -> np.ndarray:
    """Simultaneous KL lower-bound decision criterion for non-baseline cells.

    Exactly equivalent (up to endpoint rounding) to inverted Bernoulli KL
    lower bound >= threshold. This function is ONLY for planning forecasts;
    query_lab.assess remains the final authoritative safety check.
    """
    phat = np.asarray(safety_sum, dtype=float) / counts
    limit = log(4 * counts.size * maximum / delta)
    certified = ((phat > threshold) &
                 (counts * bernoulli_kl(phat, threshold) > limit))
    certified[:, 0] = True  # No-op safety is known by simulator construction.
    return certified


def constrained_batch_voc(reward_sum: np.ndarray, safety_sum: np.ndarray,
                          counts: np.ndarray, weights: np.ndarray,
                          threshold: float, delta: float,
                          maximum: int, batch: int) -> dict:
    """Expected one-batch change in constrained posterior-best reward.

    Future safety certification is recomputed under every possible predictive
    safety count k_s; reward posterior under every possible reward count k_r.
    The two count distributions factor by an ASSUMED independent prior and
    independent outcome channels. Return signed EVSI and expected-positive
    decision improvement separately; a negative signed result may reflect
    discovering that an apparently certified action loses certification.
    """
    rewards = np.asarray(reward_sum, dtype=float)
    safes = np.asarray(safety_sum, dtype=float)
    n = np.asarray(counts, dtype=float)
    w = np.asarray(weights, dtype=float)
    if (rewards.ndim != 2 or safes.shape != rewards.shape or n.shape != rewards.shape
            or w.shape != (rewards.shape[0],) or np.any(n < 1)
            or np.any(rewards < 0) or np.any(rewards > n)
            or np.any(safes < 0) or np.any(safes > n)
            or np.any(w < 0) or not np.isclose(w.sum(), 1.0)
            or not (0 < threshold < 1) or not (0 < delta < 1)
            or not (1 <= batch <= maximum)):
        raise ValueError("invalid C-VoC evidence, weights or configuration")

    reward_posterior = (1.0 + rewards) / (2.0 + n)
    current_values = w @ reward_posterior
    certificate_cells = certified_cells(safes, n, threshold, delta, maximum)
    safe_actions = np.all(certificate_cells, axis=0)
    safe_actions[0] = True
    current_action = int(np.argmax(np.where(safe_actions, current_values, -np.inf)))
    baseline = float(current_values[current_action])

    decision_evsi = np.zeros_like(rewards, dtype=float)
    positive_improvement = np.zeros_like(rewards, dtype=float)
    certification_probability = np.zeros_like(rewards, dtype=float)
    k = np.arange(batch + 1, dtype=float)

    for m in range(rewards.shape[0]):
        for a in range(rewards.shape[1]):
            # Only the queried action's reward and safety can change.
            rivals = [j for j in range(rewards.shape[1])
                      if j != a and safe_actions[j]]
            other_best = max((float(current_values[j]) for j in rivals),
                             default=-np.inf)
            # Other model cells' safety status is held fixed in the forecast.
            other_models_safe = bool(np.all(np.delete(certificate_cells[:, a], m)))
            if a == 0:
                future_is_certified = np.ones(batch + 1, dtype=bool)
                ps = np.array([1.0])
            elif other_models_safe:
                updated_n = n[m, a] + batch
                next_phat = (safes[m, a] + k) / updated_n
                limit = log(4 * rewards.size * maximum / delta)
                future_is_certified = ((next_phat > threshold) &
                    (updated_n * bernoulli_kl(next_phat, threshold) > limit))
                ps = beta_binomial_pmf(1 + safes[m, a],
                                       1 + n[m, a] - safes[m, a], batch)
            else:
                future_is_certified = np.zeros(batch + 1, dtype=bool)
                ps = beta_binomial_pmf(1 + safes[m, a],
                                       1 + n[m, a] - safes[m, a], batch)

            p_cert = (1.0 if a == 0 else float(np.sum(ps[future_is_certified])))
            certification_probability[m, a] = p_cert
            if not np.isfinite(other_best) and p_cert < 1.0:
                raise ValueError("requires known-safe baseline action")

            pr = beta_binomial_pmf(1 + rewards[m, a],
                                   1 + n[m, a] - rewards[m, a], batch)
            updated_mean = (1.0 + rewards[m, a] + k) / (2.0 + n[m, a] + batch)
            future_value = (current_values[a] +
                            w[m] * (updated_mean - reward_posterior[m, a]))
            if np.isfinite(other_best):
                if_certified = np.maximum(other_best, future_value)
                expected_best = (p_cert * float(np.dot(pr, if_certified))
                                 + (1.0 - p_cert) * other_best)
                gain_if_certified = np.maximum(0.0, if_certified - baseline)
                gain_if_not_certified = max(0.0, other_best - baseline)
                positive_improvement[m, a] = (
                    p_cert * float(np.dot(pr, gain_if_certified)) +
                    (1.0 - p_cert) * gain_if_not_certified)
            else:
                # a=0 and no other action certified: known-safe baseline
                # must remain certified; posterior mean martingale gives EVSI 0.
                expected_best = float(np.dot(pr, future_value))
                positive_improvement[m, a] = float(
                    np.dot(pr, np.maximum(0.0, future_value - baseline)))
            decision_evsi[m, a] = expected_best - baseline
    return {
        "decision_evsi": decision_evsi,
        "expected_positive_improvement": positive_improvement,
        "future_action_certification_probability": certification_probability,
        "current_certified_action": current_action,
        "current_posterior_value": baseline,
        "model": "independent_beta_bernoulli_one_batch_lookahead",
        "not_a_safety_certificate": True,
    }
