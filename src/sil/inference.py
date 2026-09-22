"""Approximate hidden-regime Bayesian filter with explicit Gaussian likelihood."""
import numpy as np
from .core import MODEL_STRENGTHS, SIGMA, step

def posterior(prior: np.ndarray, previous: np.ndarray,
              action: float, observation: np.ndarray) -> np.ndarray:
    """Uses a recorded simulated transition, not a fabricated mock observation.

    Clipping makes the Gaussian likelihood approximate near state boundaries.
    """
    prior = np.asarray(prior, dtype=float)
    if prior.shape != (len(MODEL_STRENGTHS),) or np.any(prior <= 0):
        raise ValueError("prior must contain positive mass for every model")
    prior = prior / prior.sum()
    switching_mass = 0.15
    predicted_prior = (1 - switching_mass) * prior + switching_mass / len(prior)
    means = np.array([step(previous, action, strength, np.zeros(2))
                      for strength in MODEL_STRENGTHS])
    residual = (observation - means) / SIGMA
    logw = np.log(predicted_prior) - 0.5 * np.sum(residual * residual, axis=1)
    logw -= np.max(logw)
    weights = np.exp(logw)
    return weights / np.sum(weights)
