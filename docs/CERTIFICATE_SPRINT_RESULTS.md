# SIL v0.7 Certificate Sprint: prospective experiments and limits

All outputs here are from finite, predeclared, **synthetic** Bernoulli Query Lab simulations, not a validation of any real-market, defense, geopolitical or autonomous decision policy. The analytically feasible-action oracle uses declared simulator probabilities that are unavailable in real-world data. The known-safe no-op is a defining assumption of THIS toy. Every actual selected non-baseline action still passes the same observed-data, time-uniform KL safety check; pilot safety and posterior means are only allocation signals.

## Critical fair-control change

The first v0.7 96-context comparison against the original small-batch uniform gave Sprint 45/96 versus small-batch uniform 31/96, at equal 576 cell queries. The apparently lower Sprint CPU in that experiment was confounded by its efficient *bulk* query execution. The required control \`uniform_bulk\` now has the same grouped simulation mechanics and makes the same selections as original uniform at the divisible-grid benchmark budgets. A test compares its complete seeded per-cell reward and safety successes with old uniform, not just a final score.

The v0.7 code also uses **lazy** per-cell and per-channel seeded random streams: only selected queries produce random outcomes. The experiment reports total measured process CPU and wall-clock time, not just a count of logical queries.

## Prospective study 1 (unchanged heuristic, unselected seeds 202623/24/25)

[GitHub run and raw JSON artifacts](https://github.com/aixaria0/strategic-intelligence-lattice/actions/runs/35796607416).

Same-query comparison: 96 contexts across four declared scenario families, budget 576 actually sampled cells for each method; timing repeated 3 times and reported as medians per context. Full-grid fixed at 1152 queries is a higher-cost accuracy reference, NOT an equal-budget comparator.

| Algorithm | Oracle agreement | Mean constrained regret | Mean measured process CPU (seconds) |
|---|---:|---:|---:|
| Small-batch uniform | 28/96 | 0.11275 | 0.003394 |
| **Uniform Bulk** | **28/96** | **0.11275** | **0.003094** |
| Certificate Sprint | 44/96 | 0.07901 | 0.003184 |
| Beta-Binomial predictive portfolio | 43/96 | 0.07860 | 0.011573 |
| Full-grid fixed at 1152 draws | 46/96 | 0.07452 | 0.003788 |

At this limited seed count, Sprint improves synthetic decision quality at equal query count, but measured CPU was about 2.9% HIGHER than bulk-uniform. The tiny millisecond CPU differences are sensitive to machine and scheduling.

Separate 32-context calibration selected Sprint at 576 queries within bulk-uniform's CALIBRATION mean CPU. On a disjoint 48-context holdout, Sprint matched the oracle 18/48 versus bulk-uniform 11/48, with mean regret 0.08204 versus 0.11419. However, Sprint consumed 0.003145 mean CPU seconds versus bulk-uniform 0.003091: about 1.8% higher CPU on holdout. The calibration acceptance did not establish actual holdout CPU dominance.

## Prospective study 2 (closed-form scheduling optimization, fresh seeds 202630/31/32)

The optimized sprint_schedule() replaces repeated water filling with an integer closed-form schedule in the common equal-pilot case. The original general schedule remains as fallback for unequal input counts. Seeds were changed in the workflow BEFORE the second comparison; no result from these seeds was used to select policy thresholds.

[GitHub run, all 96 scenario records and CPU calibration/holdout](https://github.com/aixaria0/strategic-intelligence-lattice/actions/runs/35796831377).

| Algorithm | Oracle agreement | Mean constrained regret | Mean measured process CPU (seconds) |
|---|---:|---:|---:|
| Uniform Bulk | 28/96 | 0.10584 | 0.004380 |
| Certificate Sprint | 42/96 | 0.07550 | 0.004475 |
| Predictive portfolio | 43/96 | 0.07242 | 0.015784 |
| Full-grid fixed at 1152 draws | 46/96 | 0.06675 | 0.005396 |

In the separately seeded 32-context calibration, Sprint's 576-query mean CPU was 0.004763 s versus 0.004885 s for uniform_bulk: Sprint at 576 passed the CALIBRATION budget. In a distinct 48-context holdout, Sprint matched the analytic oracle 19/48 versus uniform_bulk 13/48 and reduced mean regret from 0.10551 to 0.08412, but used mean CPU 0.004508 s versus bulk-uniform 0.004387 s (about 2.8% MORE). All conservative selections were analytically safe under all three specified model probabilities in these finite experiments; this does not establish a safety guarantee under model misspecification.

## Interpretation and release rule

Sprint is an effective **task-specific, low-overhead certificate-focused sampler** on the four synthetic families, especially the family deliberately designed around three-model safety certification. It is NOT an exact expected value-of-computation optimizer, robust proof of safety, or a universal decision engine. Neither independent CPU holdout has demonstrated mean end-to-end CPU <= bulk-uniform, even though seed-specific calibration selected equal query budgets. Noise, model fit and sampling variation prohibit turning small speed differences into a generalized speed claim.

Thus the query-lab default changes from small-batch uniform to its **decision-equivalent, faster uniform_bulk implementation**, while certificate_sprint remains an explicit experimental choice. The negative equal-CPU conclusions from v0.6 and the residual holdout CPU overrun in v0.7 are preserved.

For a publishable extension: stress the policy under reward-safety correlations, missing adversary hypotheses, no known-safe fallback, nonstationary probabilities and fixed real CPU deadlines. Measure CPU on different hardware and predeclare paired uncertainty intervals; consider a portfolio policy ONLY if end-to-end constrained decision gain survives fair implementation and model-coverage controls.
