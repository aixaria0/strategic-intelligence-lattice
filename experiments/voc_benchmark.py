"""Prespecified, negative-results-preserving C-VoC synthetic benchmark.

Three types of contextual decision problems:
- original crossed rewards and model-dependent safety;
- reward-competitive, high but varying safety probabilities;
- near-threshold safety, where conservative certification may require
  more data than any tested method receives.

Per scenario/method: identical independent query streams, analytic oracle,
real measured wall time, measured planning vs sampling time (C-VoC), actual
query count, certified/exploratory action and true safety. No market claims.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import numpy as np

from sil.query_lab import QueryConfig, make_scenario, allocate

METHODS = ("fixed", "uniform", "adaptive", "hybrid",
           "evsi_reward", "c_voc", "c_voc_governor")


def make_context(seed: int, index: int, kind: str, threshold: float):
    case = make_scenario(seed, index, threshold)
    if kind == "crossing":
        return case
    reward, safe = case.reward_prob.copy(), case.safety_prob.copy()
    rng = np.random.default_rng(np.random.SeedSequence([seed, index, 633]))
    if kind == "near_safe":
        # Designed before benchmarking; actionable opportunities sometimes
        # depend on estimating the least-safe hypothetical regime.
        for a in (1, 2):
            safe[:, a] = np.clip(
                threshold + rng.uniform(0.06, 0.20) -
                rng.uniform(0.0, 0.035, size=3), 0.05, 0.998)
        reward[:, 1] = np.clip(0.57 + rng.uniform(-0.08, 0.08) +
                               rng.normal(0, 0.02, 3), 0.05, 0.95)
        reward[:, 2] = np.clip(0.57 + rng.uniform(-0.08, 0.08) +
                               rng.normal(0, 0.02, 3), 0.05, 0.95)
    elif kind == "boundary":
        # True safety may be infinitesimally above or below the threshold.
        # A safe no-op remains available; good allocators may correctly
        # conclude that budget cannot certify a non-baseline action.
        for a in (1, 2):
            safe[:, a] = np.clip(
                threshold + rng.uniform(-0.035, 0.035) +
                rng.normal(0, 0.012, 3), 0.05, 0.998)
        reward[:, 1:] = np.clip(reward[:, 1:] + 0.08, 0.05, 0.98)
    else:
        raise ValueError("unknown scenario family")
    return replace(case, reward_prob=reward, safety_prob=safe)


def experiment(seeds: int = 12, budget: int = 576, maximum: int = 128,
               seed: int = 8800, compute_price: float = 0.0001):
    if not 2 <= seeds <= 500:
        raise ValueError("seeds must be 2..500")
    rows = []
    for kind in ("crossing", "near_safe", "boundary"):
        for i in range(seeds):
            threshold = (0.70, 0.78, 0.85)[i % 3]
            scenario_id = i + {"crossing": 0, "near_safe": 100000,
                               "boundary": 200000}[kind]
            cfg = QueryConfig(seed=seed, maximum=maximum, budget=budget,
                              threshold=threshold, compute_price=compute_price)
            case = make_context(seed, scenario_id, kind, threshold)
            results = {}
            for method in METHODS:
                start = perf_counter()
                result = allocate(case, cfg, method)
                result["wall_seconds"] = perf_counter() - start
                results[method] = result
            rows.append({"family": kind, "id": scenario_id,
                         "threshold": threshold, "oracle": case.oracle_action,
                         "results": results})

    summaries = {}
    for family in ("all", "crossing", "near_safe", "boundary"):
        cases = [r for r in rows if family == "all" or r["family"] == family]
        summaries[family] = {}
        for method in METHODS:
            m = [r["results"][method] for r in cases]
            summaries[family][method] = {
                "count": len(m),
                "mean_queries": float(np.mean([x["used"] for x in m])),
                "mean_seconds": float(np.mean([x["wall_seconds"] for x in m])),
                "oracle_agreement": float(np.mean([x["action"] == x["oracle"]
                                                   for x in m])),
                "mean_true_regret": float(np.mean([x["regret"] for x in m])),
                "unsafe_certified_count": int(sum(not x["truly_safe"] for x in m)),
                "exploratory_oracle_agreement": float(np.mean(
                    [x["exploratory_oracle_agreement"] for x in m])),
                "unsafe_exploratory_count": int(sum(
                    not x["exploratory_truly_safe"] for x in m)),
                "governor_stop_count": int(sum(
                    x["stop_reason"] == "governor_one_step_value_below_price"
                    for x in m)),
                "planning_seconds_mean": float(np.mean(
                    [x.get("planning_seconds", 0) for x in m])),
                "sampling_seconds_mean": float(np.mean(
                    [x.get("sampling_seconds", 0) for x in m])),
            }
    return {
        "domain": "synthetic independent Bernoulli contexts, NOT live decisions",
        "method_notes": {
            "fixed": "full grid 9*maximum, larger budget (accuracy ceiling)",
            "uniform": "equal constrained budget, research default",
            "c_voc": "paired reward+safety Beta-Binomial one-step predictive decision value",
            "c_voc_governor": "same heuristic with opt-in compute price and minimum evidence gate; can prematurely stop",
        },
        "budget": budget, "maximum": maximum, "seeds_per_family": seeds,
        "compute_price": compute_price, "summary": summaries,
        "oracle_action_counts": {
            fam: {str(a): sum(x["oracle"] == a for x in rows if x["family"] == fam)
                  for a in range(3)}
            for fam in ("crossing", "near_safe", "boundary")
        },
        "raw": rows,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=12)
    p.add_argument("--budget", type=int, default=576)
    p.add_argument("--maximum", type=int, default=128)
    p.add_argument("--seed", type=int, default=8800)
    p.add_argument("--compute-price", type=float, default=0.0001)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    report = experiment(a.seeds, a.budget, a.maximum, a.seed, a.compute_price)
    print(json.dumps({"oracle_action_counts": report["oracle_action_counts"],
                      "summary": report["summary"]}, indent=2))
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
