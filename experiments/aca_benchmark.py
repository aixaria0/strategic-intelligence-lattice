"""Reproducible fixed-versus-adaptive computation benchmark.

Use independent held-out random seeds and the same synthetic starting state and
fixed reward. Report actual rollout counts AND wall time: adaptive re-analysis
can cost more CPU even when it simulates fewer trajectories.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from sil.allocation import Allocation, evaluate_adaptive
from sil.core import ACTIONS, MODEL_STRENGTHS, Config
from sil.engine import Tournament, evaluate


def choose(rows: list[dict]) -> int:
    feasible = [i for i, row in enumerate(rows) if row["feasible"]]
    if not feasible:
        return 0
    return max(feasible, key=lambda i: rows[i]["score"])


def benchmark(seeds: int = 12, trials: int = 128, horizon: int = 8,
              holdout_trials: int = 512, budget: int | None = None) -> dict:
    if not 1 <= seeds <= 200:
        raise ValueError("seeds must be 1..200")
    if holdout_trials < 16:
        raise ValueError("holdout_trials must be >=16")
    records = []
    for i in range(seeds):
        seed = 5000 + i
        initial = (0.24 + 0.06 * (i % 5), 0.42 + 0.06 * (i % 5))
        cfg = Config(trials=trials, horizon=horizon, seed=seed,
                     initial_state=initial)
        agent = Tournament(cfg, num_agents=2, evolve=False).agents[0]
        agent.prior = np.array([0.15, 0.25, 0.60]) if i % 2 else np.ones(3) / 3
        agent.alpha = 0.12 + 0.13 * (i % 3)
        agent.beta = 0.03 + 0.10 * (i % 3)
        cap = len(ACTIONS) * len(MODEL_STRENGTHS) * cfg.trials
        t0 = perf_counter()
        fixed = evaluate(agent, cfg, 1)
        fixed_seconds = perf_counter() - t0
        t0 = perf_counter()
        adaptive, diagnostics = evaluate_adaptive(
            agent, cfg, 1, Allocation(budget=cap if budget is None else budget))
        adaptive_seconds = perf_counter() - t0
        heldout_cfg = Config(trials=holdout_trials, horizon=horizon,
                             seed=seed + 100_000, initial_state=initial)
        heldout = evaluate(agent, heldout_cfg, 1)
        oracle = choose(heldout)
        def result(rows: list[dict], spent: int, seconds: float) -> dict:
            selected = choose(rows)
            heldout_valid = heldout[selected]["feasible"]
            regret = (float(max(0.0, heldout[oracle]["score"] -
                                heldout[selected]["score"]))
                      if heldout_valid else None)
            return {"selected_action": rows[selected]["action"],
                    "rollouts": spent,
                    "wall_seconds": seconds,
                    "holdout_feasible": bool(heldout_valid),
                    "holdout_score": float(heldout[selected]["score"]),
                    "holdout_selection_regret": regret,
                    "matches_holdout_oracle": bool(selected == oracle)}
        records.append({
            "seed": seed, "initial_state": initial, "holdout_oracle_action": heldout[oracle]["action"],
            "fixed": result(fixed, cap, fixed_seconds),
            "adaptive": {
                **result(adaptive, diagnostics["rollouts_used"], adaptive_seconds),
                "model_trials": diagnostics["model_trials"],
                "stop_reason": diagnostics["stop_reason"],
            },
        })
    def average(method: str, metric: str) -> float:
        values = [r[method][metric] for r in records if r[method][metric] is not None]
        return float(np.mean(values)) if values else float("nan")
    return {
        "protocol": "same maximum action-rollout budget; same seed/state/prior and independent held-out shocks",
        "confidence": "exploratory descriptive results only; no statistical guarantee, theorem or real market edge",
        "seeds": seeds, "trials_per_model_action_fixed": trials,
        "holdout_trials_per_model_action": holdout_trials,
        "round_id": 1, "max_budget_per_method": cap if budget is None else budget,
        "summary": {
            "fixed_mean_rollouts": average("fixed", "rollouts"),
            "adaptive_mean_rollouts": average("adaptive", "rollouts"),
            "fixed_mean_seconds": average("fixed", "wall_seconds"),
            "adaptive_mean_seconds": average("adaptive", "wall_seconds"),
            "fixed_oracle_agreement": average("fixed", "matches_holdout_oracle"),
            "adaptive_oracle_agreement": average("adaptive", "matches_holdout_oracle"),
            "fixed_holdout_feasible_fraction": average("fixed", "holdout_feasible"),
            "adaptive_holdout_feasible_fraction": average("adaptive", "holdout_feasible"),
            "fixed_mean_regret_on_feasible": average("fixed", "holdout_selection_regret"),
            "adaptive_mean_regret_on_feasible": average("adaptive", "holdout_selection_regret"),
        },
        "records": records,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=12)
    p.add_argument("--trials", type=int, default=128)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--holdout", type=int, default=512)
    p.add_argument("--budget", type=int, default=None)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = benchmark(args.seeds, args.trials, args.horizon,
                       args.holdout, args.budget)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
