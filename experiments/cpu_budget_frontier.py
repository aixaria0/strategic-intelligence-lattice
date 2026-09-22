"""Predeclared CPU-frontier CALIBRATION and INDEPENDENT holdout check.

Crucial honesty: equal simulated query counts are not equal CPU time.
Calibrate an affordable plan budget using separate context seeds and
process_time() on the same runner as a BULK uniform baseline. If even the
minimum pilot cannot meet the uniform CPU budget, declare NO CPU-matched
candidate. Do not re-label a slower method as efficient.

This is a finite descriptive experiment; millisecond CPU timings can vary
by system load, Python version and process, so always retain raw records.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import replace
from pathlib import Path
from time import process_time
import numpy as np
from certificate_portfolio_benchmark import context, FAMILIES
from sil.query_lab import QueryConfig, allocate

CANDIDATES = (144, 288, 432, 576)
METHODS = ("certificate_portfolio", "certificate_targeted", "certificate_sprint")


def measure(case, cfg, method, repeats):
    # Warm reproducible code paths; do not include shared packaging/import cost.
    allocate(case, cfg, method)
    samples = []
    item = None
    for _ in range(repeats):
        t = process_time()
        item = allocate(case, cfg, method)
        samples.append(process_time() - t)
    return float(np.median(samples)), item


def evaluate_slice(seed, n, budget, maximum, repeats):
    records = []
    for family_idx, family in enumerate(FAMILIES):
        for i in range(n):
            identifier = family_idx * 100_000 + i
            threshold = (0.70, 0.78, 0.85)[i % 3]
            case = context(seed, identifier, family, threshold)
            base_cfg = QueryConfig(maximum=maximum, budget=budget,
                                   threshold=threshold, seed=seed)
            base_cpu, base = measure(case, base_cfg, "uniform_bulk", repeats)
            variants = {}
            for method in METHODS:
                for size in CANDIDATES:
                    if size > budget or size > 9 * maximum:
                        continue
                    cpu, selected = measure(
                        case, replace(base_cfg, budget=size), method, repeats)
                    variants[f"{method}@{size}"] = {
                        "median_cpu_seconds": cpu,
                        "oracle_match": selected["action"] == selected["oracle"],
                        "used": selected["used"],
                        "regret": selected["regret"],
                        "safe": selected["truly_safe"],
                    }
            records.append({
                "family": family, "scenario": identifier,
                "baseline": {"median_cpu_seconds": base_cpu,
                             "oracle_match": base["action"] == base["oracle"],
                             "used": base["used"], "regret": base["regret"],
                             "safe": base["truly_safe"]},
                "variants": variants,
            })
    return records


def summarize(records, chosen):
    values = [row[chosen] if chosen == "baseline"
              else row["variants"][chosen] for row in records]
    return {
        "contexts": len(values),
        "mean_cpu_seconds": float(np.mean([x["median_cpu_seconds"] for x in values])),
        "oracle_matches": int(sum(x["oracle_match"] for x in values)),
        "mean_queries": float(np.mean([x["used"] for x in values])),
        "mean_regret": float(np.mean([x["regret"] for x in values])),
        "unsafe_certified_count": int(sum(not x["safe"] for x in values)),
    }


def study(calibration=8, holdout=12, budget=576, maximum=128,
          repeats=3, calibration_seed=202613, holdout_seed=202614):
    if calibration < 2 or holdout < 2 or not 1 <= repeats <= 10:
        raise ValueError("invalid sample settings")
    training = evaluate_slice(calibration_seed, calibration, budget, maximum, repeats)
    baseline = summarize(training, "baseline")
    calibration_summary = {f"{method}@{size}":
       summarize(training, f"{method}@{size}")
       for method in METHODS for size in CANDIDATES
       if size <= budget and size <= 9 * maximum}
    choices = {}
    for method in METHODS:
        eligible = [size for size in CANDIDATES
                    if f"{method}@{size}" in calibration_summary
                    and calibration_summary[f"{method}@{size}"]["mean_cpu_seconds"]
                    <= baseline["mean_cpu_seconds"]]
        choices[method] = max(eligible) if eligible else None
    # Independently seeded holdout; budget selection MUST NOT use its outcomes.
    heldout = evaluate_slice(holdout_seed, holdout, budget, maximum, repeats)
    holdout_results = {"uniform_bulk": summarize(heldout, "baseline")}
    for method, size in choices.items():
        if size is None:
            holdout_results[method] = {
                "status": "NO_CPU_MATCH_AT_OR_ABOVE_PILOT",
                "interpretation": "do not claim equal-CPU superiority",
            }
        else:
            holdout_results[method] = {
                "status": "CALIBRATION_CPU_MATCHED",
                "calibrated_max_queries": size,
                "result": summarize(heldout, f"{method}@{size}"),
            }
    return {
        "scope": "same-runner finite synthetic calibration and independent holdout",
        "baseline_budget": budget, "per_cell_maximum": maximum,
        "timing_repetitions": repeats,
        "calibration_seed": calibration_seed, "holdout_seed": holdout_seed,
        "calibration": {"uniform_bulk": baseline, "candidate_budgets": calibration_summary,
                        "chosen_budgets": choices},
        "holdout": holdout_results,
        "limitations": [
            "CPU budgets are calibrated on mean observed process CPU, not a hard per-scenario deadline.",
            "Cold-start import/compilation not included; per-millisecond timing noise remains.",
            "No budget match when pilot cost already exceeds baseline must be reported, not imputed.",
            "Synthetic distributions and fixed model family are not external deployment evidence.",
        ],
        "raw_calibration": training, "raw_holdout": heldout,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--calibration", type=int, default=8)
    p.add_argument("--holdout", type=int, default=12)
    p.add_argument("--budget", type=int, default=576)
    p.add_argument("--maximum", type=int, default=128)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    report = study(a.calibration, a.holdout, a.budget, a.maximum, a.repeats)
    print(json.dumps({"calibration": report["calibration"],
                      "holdout": report["holdout"]}, indent=2))
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
