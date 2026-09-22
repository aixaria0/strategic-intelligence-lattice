"""Low-overhead certificate sprint: one pilot decision, a few bulk queries.

This is a deliberately inexpensive baseline for the equal-CPU bottleneck.
It is NOT posterior-predictive EVSI, not a statistical safety certificate,
and it does not stop early merely because a toy pilot sees no opportunity.
Only the final independent KL gate can authorize a non-baseline action.
"""
from __future__ import annotations
import numpy as np


def choose_certificate_sprint(reward_sum, safety_sum, counts, weights,
                              threshold: float) -> dict | None:
    r, s, n = map(np.asarray, (reward_sum, safety_sum, counts))
    w = np.asarray(weights, dtype=float)
    if (r.shape != (3, 3) or s.shape != r.shape or n.shape != r.shape
            or w.shape != (3,) or np.any(n < 1) or
            np.any(r < 0) or np.any(s < 0) or
            np.any(r > n) or np.any(s > n) or
            np.any(w < 0) or not np.isclose(w.sum(), 1.0)
            or not 0 < threshold < 1):
        raise ValueError("invalid sprint input")
    # Beta-smoothed reward estimate and raw safety success proportion are
    # just targeting signals, not a posterior safety certification.
    scores = w @ ((r + 1.) / (n + 2.))
    best = None
    for action in (1, 2):
        if float(scores[action] - scores[0]) < 0.018:
            continue
        safe_rates = s[:, action] / n[:, action]
        # A candidate that looks unsafe even with pilot draws is not an
        # efficient certificate target in this finite synthetic experiment.
        if not np.all(safe_rates >= threshold):
            continue
        # Incentivize expected reward but discourage targets with repeatedly
        # observed failures; no hard real-world risk guarantees are claimed.
        signal = float(scores[action] - scores[0]) * float(
            np.prod((s[:, action] + 1.) / (n[:, action] + 2.)))
        if best is None or signal > best["pilot_value_signal"]:
            best = {"target_action": action,
                    "pilot_value_signal": signal,
                    "pilot_safety_rates": safe_rates.tolist(),
                    "not_a_safety_certificate": True}
    return best


def sprint_schedule(counts, target: int | None, budget: int,
                    maximum: int) -> list[tuple[int, int, int]]:
    """Bulks per-cell queries while preserving the exact total budget.

    A candidate gets up to three model-wide full-cell caps, then unused
    budget is evenly spread across other cells. If no candidate qualifies,
    the entire budget is evenly distributed. No unseen random draws.
    """
    n = np.asarray(counts, dtype=int)
    if n.shape != (3, 3) or target not in (None, 1, 2):
        raise ValueError("invalid sprint schedule")
    if np.any(n < 1) or np.any(n > maximum) or not int(n.sum()) <= budget <= 9 * maximum:
        raise ValueError("invalid budget or counts")
    proposed = n.copy()
    remaining = budget - int(n.sum())
    if target is not None:
        cells = [(m, target) for m in range(3)]
        eligible = min(remaining, sum(maximum - int(n[m, target]) for m in range(3)))
        quotient, rest = divmod(eligible, 3)
        for i, (m, a) in enumerate(cells):
            amount = min(maximum - int(proposed[m, a]), quotient + int(i < rest))
            proposed[m, a] += amount
            remaining -= amount
    while remaining:
        eligible_cells = [(m, a) for m in range(3) for a in range(3)
                          if proposed[m, a] < maximum]
        if not eligible_cells:
            break
        # Exact balanced water filling over nine integer-count cells, with
        # at most 9 * maximum accounting operations in the worst case.
        level = min(proposed[m, a] for m, a in eligible_cells)
        group = [(m, a) for m, a in eligible_cells if proposed[m, a] == level]
        next_level = min((proposed[m, a] for m, a in eligible_cells
                          if proposed[m, a] > level), default=maximum)
        increment = min(next_level - level, remaining // len(group))
        if increment:
            for m, a in group:
                proposed[m, a] += increment
            remaining -= increment * len(group)
        else:
            for m, a in group[:remaining]:
                proposed[m, a] += 1
                remaining -= 1
    return [(m, a, int(proposed[m, a] - n[m, a]))
            for m in range(3) for a in range(3)
            if proposed[m, a] > n[m, a]]
