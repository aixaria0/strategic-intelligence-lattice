"""Prespecified v0.6 multibatch certificate budget experiment.

Four independent synthetic scenario families, including a deliberately
certifiable candidate family where one-step safety EVSI is initially zero
but multibatch jointly acquired evidence may unlock it.

Fixed uses FULL GRID and is explicitly a more costly accuracy reference.
All remaining policies share exact cell-query caps and identical potential
outcome streams. Wall time includes all planner + sampling overhead.
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter, process_time
import numpy as np
from sil.query_lab import QueryConfig, allocate, make_scenario

METHODS = ("fixed", "uniform", "uniform_bulk", "adaptive", "c_voc", "certificate_portfolio", "certificate_targeted", "certificate_sprint")
FAMILIES = ("crossing", "near_safe", "boundary", "certificate_ready")


def context(seed, identifier, family, threshold):
    case = make_scenario(seed, identifier, threshold)
    if family == "crossing":
        return case
    rng = np.random.default_rng(np.random.SeedSequence([seed, identifier, 361]))
    reward = case.reward_prob.copy()
    safety = case.safety_prob.copy()
    if family == "certificate_ready":
        # Independent, predeclared high-safety alternatives. A few dozen
        # samples in each model can jointly certify the valuable action.
        preferred = 1 if identifier % 2 == 0 else 2
        reward[:, preferred] = np.clip(0.70 + rng.normal(0, 0.013, 3), 0.55, 0.90)
        reward[:, 3 - preferred] = np.clip(0.43 + rng.normal(0, 0.01, 3), 0.05, 0.55)
        safety[:, preferred] = np.clip(0.995 - rng.uniform(0, 0.012, 3), 0.90, 0.999)
        safety[:, 3 - preferred] = 0.55
    elif family == "near_safe":
        reward[:, 1] = np.clip(0.59 + rng.normal(0, .025, 3), .05, .95)
        reward[:, 2] = np.clip(0.59 + rng.normal(0, .025, 3), .05, .95)
        for a in (1, 2):
            safety[:, a] = np.clip(
                threshold + rng.uniform(.06, .20) - rng.uniform(0, .035, 3),
                .05, .998)
    elif family == "boundary":
        reward[:, 1:] = np.clip(reward[:, 1:] + .07, .05, .98)
        for a in (1, 2):
            safety[:, a] = np.clip(
                threshold + rng.uniform(-.035, .035) + rng.normal(0, .012, 3),
                .05, .998)
    else:
        raise ValueError("unknown scenario family")
    return replace(case, reward_prob=reward, safety_prob=safety)


def benchmark(seeds=8, maximum=128, budget=576, seed=202609, repeats=1):
    if not (2 <= seeds <= 100 and 1 <= repeats <= 10):
        raise ValueError("seeds 2..100, repeats 1..10")
    cfg = QueryConfig(maximum=maximum, budget=budget, seed=seed)
    records = []
    for family_idx, family in enumerate(FAMILIES):
        for index in range(seeds):
            identifier = family_idx * 100_000 + index
            threshold = (0.70, 0.78, 0.85)[index % 3]
            case = context(seed, identifier, family, threshold)
            run_cfg = replace(cfg, threshold=threshold)
            outcomes = {}
            for method in METHODS:
                timings = []
                cpu_timings = []
                values = []
                for repeat in range(repeats):
                    started = perf_counter()
                    cpu_started = process_time()
                    item = allocate(case, run_cfg, method)
                    cpu_timings.append(process_time() - cpu_started)
                    timings.append(perf_counter() - started)
                    values.append(item)
                first = values[0]
                for other in values[1:]:
                    for k in ("action", "counts", "used", "regret", "truly_safe"):
                        if first[k] != other[k]:
                            raise AssertionError(f"nonreproducible seeded output: {method}.{k}")
                first["median_wall_seconds"] = float(np.median(timings))
                first["median_cpu_seconds"] = float(np.median(cpu_timings))
                first["wall_seconds_by_repeat"] = timings
                first["cpu_seconds_by_repeat"] = cpu_timings
                outcomes[method] = first
            records.append({
                "family": family, "id": identifier, "oracle": case.oracle_action,
                "threshold": threshold, "model_weights": list(case.posterior),
                "outcomes": outcomes,
            })
    def summary(rows):
        result = {}
        for method in METHODS:
            vals = [row["outcomes"][method] for row in rows]
            result[method] = {
                "count": len(vals),
                "oracle_agreement": int(sum(x["action"] == x["oracle"] for x in vals)),
                "mean_true_regret": float(np.mean([x["regret"] for x in vals])),
                "mean_queries": float(np.mean([x["used"] for x in vals])),
                "median_wall_seconds": float(np.median([x["median_wall_seconds"] for x in vals])),
                "mean_wall_seconds": float(np.mean([x["median_wall_seconds"] for x in vals])),
                "mean_cpu_seconds": float(np.mean([x["median_cpu_seconds"] for x in vals])),
                "unsafe_certified_count": int(sum(not x["truly_safe"] for x in vals)),
                "portfolio_count": int(sum(len(x.get("portfolios", [])) for x in vals)),
                "mean_planning_seconds": float(np.mean(
                    [x.get("planning_seconds", 0) for x in vals])),
            }
        return result
    report = {
        "protocol": "synthetic independent Bernoulli, analytic oracle, fixed predeclared query streams",
        "fixed_full_grid_budget": 9 * maximum,
        "equal_query_budget_for_other_methods": budget,
        "seeds_per_family": seeds, "repeats_for_wall_time": repeats,
        "oracle_distribution": {family: {str(a): sum(
            row["family"] == family and row["oracle"] == a for row in records
        ) for a in range(3)} for family in FAMILIES},
        "summary_all": summary(records),
        "summary_by_family": {f: summary([row for row in records if row["family"] == f])
                              for f in FAMILIES},
        "raw": records,
        "limitations": [
            "The certificate planner uses a finite coarse grid of future allocations, not exact optimal multistep EVSI.",
            "Synthetic independent Beta-Bernoulli cell likelihoods may misrepresent external observations.",
            "Fixed has a larger query budget; equal query counts are not equal wall/CPU costs.",
            "A returned unsafe exploratory action is never equivalent to an authorized certified choice.",
            "Run on several machines and predeclared holdout scenario families before claiming general value.",
        ],
    }
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--maximum", type=int, default=128)
    p.add_argument("--budget", type=int, default=576)
    p.add_argument("--seed", type=int, default=202609)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    report = benchmark(a.seeds, a.maximum, a.budget, a.seed, a.repeats)
    print(json.dumps({"oracles": report["oracle_distribution"],
                      "all": report["summary_all"],
                      "by_family": report["summary_by_family"]}, indent=2))
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
