"""Vectorized synthetic stochastic simulation and empirical binned terminal entropy.

Terminal state statistics are not an attractor proof; uncertainty alone is not value.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

MODEL_STRENGTHS = np.array([0.0, 0.5, 1.0], dtype=float)
ACTIONS = np.array([0.0, 0.5, 1.0], dtype=float)
SIGMA = np.array([0.025, 0.012], dtype=float)

@dataclass(frozen=True)
class Config:
    trials: int = 128
    horizon: int = 8
    seed: int = 2026
    min_safe_probability: float = 0.80
    initial_state: tuple[float, float] = (0.48, 0.60)

    def __post_init__(self) -> None:
        if not 16 <= self.trials <= 100_000:
            raise ValueError("trials must be 16..100000")
        if not 1 <= self.horizon <= 256:
            raise ValueError("horizon must be 1..256")
        if not 0 <= self.min_safe_probability <= 1:
            raise ValueError("min_safe_probability must be 0..1")
        if len(self.initial_state) != 2 or any(not 0 <= v <= 1 for v in self.initial_state):
            raise ValueError("initial_state must be two unit-interval values")

def step(states: np.ndarray, action: float, strength: float, noise: np.ndarray) -> np.ndarray:
    drift = np.array([0.015 + 0.080 * action - 0.052 * strength,
                      0.020 - 0.065 * action - 0.012 * strength])
    return np.clip(states + drift + noise, 0.0, 1.0)

def rollout(initial: np.ndarray, action: float, strength: float,
            shocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """shocks: (trials,horizon,2); shared between candidate actions."""
    s = np.broadcast_to(initial, (shocks.shape[0], 2)).copy()
    minimum_reserve = s[:, 1].copy()
    for t in range(shocks.shape[1]):
        s = step(s, action, strength, shocks[:, t, :])
        minimum_reserve = np.minimum(minimum_reserve, s[:, 1])
    return s, minimum_reserve

def discrete_entropy(points: np.ndarray, bins: int = 5) -> float:
    """Normalized empirical Shannon entropy of 2D terminal histogram."""
    if len(points) == 0:
        return 0.0
    counts, _, _ = np.histogram2d(points[:, 0], points[:, 1],
                                   bins=bins, range=((0, 1), (0, 1)))
    p = counts.ravel()
    p = p[p > 0] / len(points)
    return float(-np.sum(p * np.log(p)) / np.log(bins * bins))

def reward(terminal: np.ndarray, action: float) -> np.ndarray:
    """External fixed synthetic reward, independent of policy's selection weights."""
    return terminal[:, 0] + 0.35 * terminal[:, 1] - 0.040 * action

def make_shocks(cfg: Config, round_id: int, model_id: int) -> np.ndarray:
    ss = np.random.SeedSequence([cfg.seed, round_id, model_id, 1729])
    return np.random.default_rng(ss).normal(
        0, SIGMA, size=(cfg.trials, cfg.horizon, 2))
