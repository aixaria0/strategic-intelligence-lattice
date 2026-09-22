"""Analytic-oracle budget comparison on heterogeneous, crossed synthetic payoffs.

Queries are independent (regime, action) cells; the four schedulers use identical
seeded potential outcome streams per scenario. A full-grid fixed comparator
consumes its full cap, while uniform, random, adaptive receive EQUAL smaller
budgets. A known expected-value and safety oracle avoids test data reuse.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from sil.query_lab import QueryConfig, make_scenario, allocate


METHODS = ("fixed", "uniform", "random", "adaptive", "hybrid")


def run(seeds=48, budget=720, maximum=256, seed=2026):
    if not 2 <= seeds <= 1000:
        raise ValueError("seed count must be 2..1000")
    records = []
    for index in range(seeds):
        threshold = (0.72, 0.78, 0.84)[index % 3]
        cfg = QueryConfig(budget=budget, maximum=maximum,
                          threshold=threshold, seed=seed)
        scenario = make_scenario(seed, index, threshold)
        results = {}
        for method in METHODS:
            start = perf_counter()
            result = allocate(scenario, cfg, method)
            result["wall_seconds"] = perf_counter() - start
            results[method] = result
        records.append({
            "id": index, "oracle_action": scenario.oracle_action,
            "demand": scenario.demand, "stress": scenario.stress,
            "safety_threshold": threshold, "true_regime": scenario.true_regime,
            "posterior": list(scenario.posterior),
            "results": results,
        })

    summary = {}
    for method in METHODS:
        values = [row["results"][method] for row in records]
        summary[method] = {
            "mean_rollouts": float(np.mean([x["used"] for x in values])),
            "mean_wall_seconds": float(np.mean([x["wall_seconds"] for x in values])),
            "oracle_agreement": float(np.mean([x["action"] == x["oracle"] for x in values])),
            "mean_true_regret": float(np.mean([x["regret"] for x in values])),
            "unsafe_choice_count": int(sum(not x["truly_safe"] for x in values)),
            "certificate_count": int(sum(x["certificate"] for x in values)),
            "fallback_count": int(sum(x["action"] == 0 for x in values)),
            "exploratory_oracle_agreement": float(np.mean(
                [x["exploratory_oracle_agreement"] for x in values])),
            "exploratory_unsafe_count": int(sum(
                not x["exploratory_truly_safe"] for x in values)),
            "exploratory_fallback_count": int(sum(
                x["exploratory_action"] == 0 for x in values)),
            "exploratory_regret_given_safe": (
                float(np.mean([x["exploratory_regret_if_safe"] for x in values
                               if x["exploratory_regret_if_safe"] is not None]))
                if any(x["exploratory_regret_if_safe"] is not None for x in values)
                else None),
        }
    return {
        "domain": "synthetic crossed-payoff Bernoulli models only",
        "fixed_budget": 9 * maximum,
        "equal_budget_for_adaptive_hybrid_uniform_random": budget,
        "num_scenarios": seeds,
        "oracle_action_counts": {
            str(a): sum(row["oracle_action"] == a for row in records)
            for a in (0, 1, 2)
        },
        "summary": summary,
        "records": records,
        "limitations": [
            "The oracle evaluates expected reward under the specified posterior and safety in ALL declared models.",
            "Model correctness and iid Bernoulli draws are assumptions; no real-world external validation.",
            "Uniform cell-wise Hoeffding bound may reject truly safe actions, especially at small budgets.",
            "Equal rollout count is not equal measured CPU; report both and repeat on identical hardware.",
            "The allocator is a heuristic, not exact Bayesian value-of-information optimization.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=48)
    parser.add_argument("--budget", type=int, default=720)
    parser.add_argument("--maximum", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.seeds, args.budget, args.maximum, args.seed)
    print(json.dumps({"oracle_action_counts": result["oracle_action_counts"],
                      "summary": result["summary"]}, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
