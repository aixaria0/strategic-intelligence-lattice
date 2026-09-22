# ACA v0.2 empirical checkpoint — September 2026

This record reports GitHub Actions **exploratory synthetic** results, not an efficiency theorem or real-world value discovery. Timing is hardware/scheduler dependent and single-run measurements are not confidence intervals.

## Reproduction

Code path: `src/sil/allocation.py`. Protocols: `experiments/aca_benchmark.py` (12 original synthetic starting states) and `experiments/aca_stress.py` (24 heterogeneous synthetic states, opponent priors, alpha/beta values and safety thresholds). In both, a separate holdout simulation seed evaluates the selected action. Both policies have the *same maximum* action-rollout budget; the adaptive policy may stop early. Benchmark run: https://github.com/aixaria0/strategic-intelligence-lattice/actions/runs/35786813011 (JSON artifacts are attached to the GitHub run).

## Results of the current running-moment algorithm

| Scenario | Fixed mean action-rollouts | Adaptive mean action-rollouts | Fixed mean elapsed seconds | Adaptive mean elapsed seconds | Agreement with holdout choice, both |
|---|---:|---:|---:|---:|---:|
| 12 original scenarios; 128/model/action | 1152 | 624 | 0.00459 | 0.00471 | 12/12 |
| 12 original scenarios; 256/model/action | 2304 | 916 | 0.00502 | 0.00611 | 12/12 |
| 24 heterogeneous states/priors; 128/model/action | 1152 | 958 | 0.00337 | 0.00669 | 24/24 |

Across these narrow testbeds the two strategies selected the same action as the sample-based holdout oracle for every scenario tested. That apparent agreement does **not** establish that the experiment is challenging: the current synthetic additive transition often makes actions easy to distinguish. Observed regret of zero in these cases is not proof of a general advantage.

For original 128-case budget, mean adaptive rollout use was about 45.8% lower, but average elapsed time was comparable; the original 256-case used about 60.2% fewer action-rollouts while taking longer. The heterogeneous case used about 16.8% fewer rollouts but roughly twice the measured elapsed time. **Do not claim ACA already improves end-to-end compute efficiency or decision quality in general.**

## Why additional research is needed

- Runtime overhead includes Python loops, decision-uncertainty updates and repeated stochastic sampling; a path count alone is not a CPU or energy metric.
- Synthetic reward may have a dominant feasible action in most settings. Define precommitted ambiguous/crossing action-response scenarios, contested risk boundaries and model uncertainty that actually change the recommended action.
- Use a large independent holdout and paired seeds across policies; compare both equal rollout counts and equal measured CPU budgets.
- Report confidence intervals across many seeds and distinct problem instances, safety false-negative rates, and whether the posterior/model family is misspecified.
- The current decision-gap stopping rule and risk gate are heuristics without finite-sample guarantees.
- More speculative layers (MCTS, exact Bayesian EVSI, distributed workers) should be introduced only when benchmarks can detect a benefit over simpler methods.
