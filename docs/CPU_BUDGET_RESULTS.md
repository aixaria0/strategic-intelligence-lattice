# v0.6 CPU-budget calibrated comparison — negative result

Source workflow with complete per-context, per-method calibration and disjoint holdout JSON: https://github.com/aixaria0/strategic-intelligence-lattice/actions/runs/35795258103

## Protocol

- Synthetic Query Lab's *lazy* per-cell/per-channel seeded reward/safety Bernoulli generators: only selected queries are actually simulated. Candidate policies see identical potential outcome prefixes.
- Baseline: uniform at 576 cell queries per scenario, maximum 128 per cell, pilot 16 per cell.
- Alternative budgets prespecified BEFORE looking at holdout: 144 (pilot only), 288, 432, 576 cell queries. Methods: multi-query predictive certificate_portfolio and cheaper Beta-smoothed certificate_targeted.
- Calibrate feasible budgets using 32 contexts across four predeclared scenario families at seed 202611. For each method/budget, measure CPU via process_time() with 3 repeated same-seed executions, report median per context and mean across contexts.
- Pick the LARGEST candidate budget whose calibration mean process CPU is no greater than uniform baseline. No holdout metric is allowed to change this choice. If none qualifies, the script reports NO_CPU_MATCH instead of pretending equal CPU.
- Evaluate the calibration-selected budgets on 48 distinct contexts with holdout seed 202612; report independently measured CPU, analytic feasible-oracle agreement, true constrained regret, query count and safety.

## Calibration result (32 contexts)

Uniform at 576: mean process CPU 0.00425 s/context. Predictive portfolio at 144: 0.00392 s; at 288: 0.00886 s; at 432: 0.01154 s; at 576: 0.01508 s. Cheap targeted at 144: 0.00376 s; at 288: 0.00718 s; at 432: 0.00805 s; at 576: 0.00842 s. For BOTH alternatives, the largest matched CPU budget was just 144, i.e. mandatory all-cell pilot. CPU numbers are noisy and reflect this GitHub runner, not a hardware-independent law.

## Disjoint holdout result (48 contexts)

| Method | Calibrated query budget | Mean measured process CPU | Analytic feasible-oracle agreement | Mean true constrained regret | Unsafe certified choices |
|---|---:|---:|---:|---:|---:|
| Uniform | 576 | 0.00416 s | 12/48 | 0.11229 | 0 |
| Multi-query predictive portfolio | 144 | 0.00380 s | 4/48 | 0.14782 | 0 |
| Cheap certificate-targeted | 144 | 0.00375 s | 4/48 | 0.14782 | 0 |

**Conclusion:** v0.6 does NOT achieve a better constrained decision at equal calibrated CPU cost, despite improvements in the 576-query matched-query experiments. Early planning overhead forces the CPU-matched variants to the all-cell pilot, which cannot accumulate enough evidence to certify non-baseline alternatives. Neither a single green CI check nor the purpose-built certificate-ready scenarios override this failure. The original fixed/full-grid comparisons remain cost-unequal accuracy ceilings, not evidence of efficiency.

## Exact limitations

CPU selection is based on *mean* measured process time in one calibration distribution and one runner, not a hard real-time per-context CPU cap or robust cross-device benchmark. A candidate that passes calibration can exceed the baseline budget on individual contexts. Time measurements are on millisecond-scale routines and include cached/warm code paths; package import and cold startup are not included. Synthetic model probabilities, calibration and holdout scenario families may all share structural assumptions. The Bayesian certificate forecasts need not be calibrated to the scenario generation process; frequentist final safety checking remains separate. No market trading or geopolitical prediction validity follows.

## Engineering implication

Further complexity in the planner is not justified by equal-CPU evidence. Maintain uniform as the research default. Investigate faster certificate-state caching, sparse per-cell sufficient statistics, vectorized predictive tail probabilities and amortized compilation only after profiling. If these cannot recover enough CPU budget for genuinely multi-batch evidence collection, preserve the negative result. Multi-stage MCTS, distributed workers and LLM calls would add overhead before solving this core issue.
