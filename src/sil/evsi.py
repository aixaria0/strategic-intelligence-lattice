"""Exact batch expected sample information for a *reward-only* Beta-Bernoulli model.

The term exact means exact posterior predictive enumeration under independent
Beta(1,1) reward-probability priors and a fixed externally observed model
posterior. It is NOT exact VOI for chance constraints, model uncertainty,
sequential control, or the underlying real world.
"""
from __future__ import annotations
from math import exp, lgamma
import numpy as np


def beta_binomial_pmf(alpha: float, beta: float, trials: int) -> np.ndarray:
    """P(K=k) for a block of trials conditional on a Beta(alpha,beta) belief."""
    if not alpha > 0 or not beta > 0 or not 1 <= trials <= 4096:
        raise ValueError("alpha,beta must be positive and trials must be 1..4096")
    shared = (lgamma(alpha + beta) - lgamma(alpha) - lgamma(beta)
              + lgamma(trials + 1))
    logs = []
    for k in range(trials + 1):
        logp = (shared - lgamma(k + 1) - lgamma(trials - k + 1)
                + lgamma(alpha + k) + lgamma(beta + trials - k)
                - lgamma(alpha + beta + trials))
        logs.append(logp)
    logs = np.asarray(logs, dtype=float)
    probability = np.exp(logs - logs.max())
    return probability / probability.sum()


def reward_batch_evsi(reward_success: np.ndarray, counts: np.ndarray,
                      model_weights: np.ndarray, batch: int) -> np.ndarray:
    """Exact expected one-batch improvement in posterior optimal REWARD.

    Current value: max_a sum_m w_m E[p_{m,a}|D].
    A query observes K additional Bernoulli rewards from one cell (m,a);
    K follows Beta-Binomial(batch,1+successes,1+failures).
    Recompute posterior action means for every K and average max-value gain.
    The query's reward draws are *simulations*, NOT external regime evidence.
    """
    success = np.asarray(reward_success, dtype=float)
    n = np.asarray(counts, dtype=float)
    weights = np.asarray(model_weights, dtype=float)
    if (success.ndim != 2 or n.shape != success.shape or
            weights.shape != (success.shape[0],) or
            np.any(n < 0) or np.any(success < 0) or np.any(success > n) or
            np.any(weights < 0) or not np.isclose(weights.sum(), 1.0)):
        raise ValueError("invalid count matrices or probability weights")
    if not 1 <= batch <= 4096:
        raise ValueError("batch must be in 1..4096")
    posterior_alpha = 1.0 + success
    posterior_beta = 1.0 + n - success
    current_mean = posterior_alpha / (posterior_alpha + posterior_beta)
    action_values = weights @ current_mean
    baseline = float(np.max(action_values))
    info = np.empty_like(success, dtype=float)
    possible_k = np.arange(batch + 1, dtype=float)
    for m in range(success.shape[0]):
        for a in range(success.shape[1]):
            predictive = beta_binomial_pmf(
                float(posterior_alpha[m, a]), float(posterior_beta[m, a]), batch)
            next_mean = ((posterior_alpha[m, a] + possible_k) /
                         (posterior_alpha[m, a] + posterior_beta[m, a] + batch))
            next_value = (action_values[a] +
                          weights[m] * (next_mean - current_mean[m, a]))
            other = max((float(value) for j, value in enumerate(action_values)
                         if j != a), default=-np.inf)
            expected_best = float(np.dot(predictive, np.maximum(next_value, other)))
            info[m, a] = max(0.0, expected_best - baseline)
    return info
