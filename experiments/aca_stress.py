"""Held-out stress protocol for ACA under heterogeneous synthetic states and priors.

Independent from the easy benchmark distribution. Descriptive metrics, no
production predictive claims or calibrated confidence statements.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from sil.core import Config
from sil.engine import Tournament, evaluate
from sil.allocation import Allocation, evaluate_adaptive


def choice(rows):
    valid = [i for i, row in enumerate(rows) if row["feasible"]]
    return max(valid, key=lambda i: rows[i]["score"]) if valid else 0


def run(seeds=24, trials=128, holdout=1024, horizon=8):
    if seeds < 1 or seeds > 200 or trials < 16 or holdout < 16:
        raise ValueError("invalid sample settings")
    records = []
    for i in range(seeds):
        seed = 90_000 + i
        rng = np.random.default_rng(np.random.SeedSequence([seed, 871]))
        state = (float(rng.uniform(0.17, 0.76)),
                 float(rng.uniform(0.21, 0.85)))
        threshold = float(rng.choice([0.60, 0.80, 0.95]))
        cfg = Config(trials=trials, horizon=horizon, seed=seed,
                     initial_state=state, min_safe_probability=threshold)
        agent = Tournament(cfg, num_agents=2, evolve=False).agents[0]
        agent.prior = rng.dirichlet(np.ones(3))
        agent.alpha = float(rng.uniform(0, 1.2))
        agent.beta = float(rng.uniform(0, 0.6))
        cap = 9 * trials

        start = perf_counter()
        fixed = evaluate(agent, cfg, 1)
        fixed_time = perf_counter() - start
        start = perf_counter()
        adaptive, report = evaluate_adaptive(agent, cfg, 1, Allocation(budget=cap))
        adaptive_time = perf_counter() - start

        independent = Config(trials=holdout, horizon=horizon, seed=seed + 1_000_000,
                             initial_state=state, min_safe_probability=threshold)
        heldout = evaluate(agent, independent, 1)
        oracle = choice(heldout)
        def characterize(rows, used, seconds):
            picked = choice(rows)
            return {
                "action": rows[picked]["action"],
                "action_rollouts": used,
                "seconds": seconds,
                "oracle_agreement": bool(picked == oracle),
                "heldout_feasible": bool(heldout[picked]["feasible"]),
                "heldout_regret": (float(max(0.0, heldout[oracle]["score"] -
                                              heldout[picked]["score"]))
                                   if heldout[picked]["feasible"] else None)
            }
        records.append({
            "seed": seed, "state": state, "threshold": threshold,
            "prior": agent.prior.tolist(), "alpha": agent.alpha, "beta": agent.beta,
            "oracle_action": heldout[oracle]["action"],
            "fixed": characterize(fixed, cap, fixed_time),
            "adaptive": {**characterize(adaptive, report["rollouts_used"], adaptive_time),
                         "model_trials": report["model_trials"],
                         "stop_reason": report["stop_reason"]}
        })

    def mean(method, field):
        values = [row[method][field] for row in records
                  if row[method][field] is not None]
        return float(np.mean(values)) if values else None

    return {
        "scope": "synthetic heterogenous-state stress test; selected models and rewards remain assumed",
        "seed_count": seeds, "trials": trials, "holdout": holdout,
        "summary": {
            method: {field: mean(method, field)
                     for field in ("action_rollouts", "seconds", "oracle_agreement",
                                   "heldout_feasible", "heldout_regret")}
            for method in ("fixed", "adaptive")
        },
        "records": records
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=24)
    p.add_argument("--trials", type=int, default=128)
    p.add_argument("--holdout", type=int, default=1024)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    report = run(a.seeds, a.trials, a.holdout, a.horizon)
    print(json.dumps(report["summary"], indent=2))
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
