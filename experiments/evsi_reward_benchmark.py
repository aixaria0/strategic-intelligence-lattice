"""Budget-matched evaluation of exact one-batch reward EVSI as a research ablation.

Reward EVSI is NOT exact chance-constrained EVSI. The certified output always
uses the same conservative safety rule; exploratory estimates are diagnostic.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np

from sil.query_lab import QueryConfig, make_scenario, allocate

METHODS = ("uniform", "adaptive", "evsi_reward", "fixed")


def run(seeds=24, maximum=256, budget=720, seed=2048):
    records = []
    for i in range(seeds):
        threshold = (0.72, 0.78, 0.84)[i % 3]
        cfg = QueryConfig(maximum=maximum, budget=budget,
                          threshold=threshold, seed=seed)
        scenario = make_scenario(seed, i, threshold)
        row = {"scenario": i, "oracle": scenario.oracle_action, "methods": {}}
        for method in METHODS:
            start = perf_counter()
            decision = allocate(scenario, cfg, method)
            decision["elapsed_seconds"] = perf_counter() - start
            row["methods"][method] = decision
        records.append(row)
    report = {}
    for method in METHODS:
        data = [row["methods"][method] for row in records]
        report[method] = {
            "average_simulation_queries": float(np.mean([x["used"] for x in data])),
            "average_elapsed_seconds": float(np.mean([x["elapsed_seconds"] for x in data])),
            "certified_oracle_agreement": float(np.mean(
                [x["action"] == x["oracle"] for x in data])),
            "exploratory_oracle_agreement": float(np.mean(
                [x["exploratory_oracle_agreement"] for x in data])),
            "unsafe_certified_count": int(sum(not x["truly_safe"] for x in data)),
            "unsafe_exploratory_count": int(sum(not x["exploratory_truly_safe"] for x in data)),
            "average_true_regret": float(np.mean([x["regret"] for x in data])),
        }
    return {
        "protocol": "predeclared synthetic 24-scenario reward-only EVSI ablation",
        "caution": "reward EVSI ignores safety information; fixed uses full grid, not equal budget",
        "scenarios": seeds, "maximum": maximum, "equal_budget": budget,
        "oracle_action_counts": {str(a): sum(row["oracle"] == a for row in records)
                                 for a in (0, 1, 2)},
        "summary": report, "records": records,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=24)
    p.add_argument("--maximum", type=int, default=256)
    p.add_argument("--budget", type=int, default=720)
    p.add_argument("--seed", type=int, default=2048)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    result = run(a.seeds, a.maximum, a.budget, a.seed)
    print(json.dumps({"oracles": result["oracle_action_counts"],
                      "summary": result["summary"]}, indent=2))
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
