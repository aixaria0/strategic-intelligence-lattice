# Multi-query certificate planning: v0.6 experimental evidence (September 2026)

## What is implemented, without stronger claims

src/sil/certificate_planner.py constructs *finite* portfolios of future query counts across the three hypothetical regime cells that must ALL establish a safety threshold for a candidate action. The Beta-Binomial predictive probability of each fixed future query count passing the same KL safety criterion is computed and multiplied across cells, conditional on independent safety models. The joint forecast is used for portfolio ranking; actual final action approval still uses query_lab.assess and time-uniform KL bounds. The planner does NOT solve an exact finite-horizon adaptive POMDP, globally optimal expected value of information, or any real-world safety problem.

src/sil/query_lab.py exposes certificate_portfolio alongside previous fixed, uniform, ACA and C-VoC modes. A second certificate_targeted mode uses a cheap Beta-smoothed plug-in sample-count target rather than enumerating predictive probabilities; it is an intentionally weaker research comparator, not a validated faster replacement. Both preserve the model-specific cell-query budget and independent final safety gate. Uniform remains the conservative default.

A deterministic unit test constructs three uncertified model cells of an otherwise profitable candidate. An additional *single* batch cannot certify ANY one of them, so the one-step constrained C-VoC is zero. A multi-query plan of additional batches across all three cells has strictly positive joint predictive probability of certification under the declared model. This proves only the existence of the indicated delay effect in the toy setup.

## Experiment A: 32 synthetic scenarios, equal 576 cell-query cap

[GitHub workflow and raw artifacts](https://github.com/aixaria0/strategic-intelligence-lattice/actions/runs/35794516565)

Under four prespecified scenario families (crossing payoff, near-safe, uncertain boundary, and purpose-built certificate-ready), the finite portfolio selected the analytic feasible-action oracle in 14/32 cases against uniform 12/32, one-step C-VoC 8/32 (initial workflow), and the full-grid fixed accuracy reference 14/32 at **double the query budget**. In the purpose-built certificate-ready family alone, portfolio was 8/8 vs uniform 6/8, and fixed 8/8. All returned certified actions were analytically safe under the specified three-model synthetic probabilities in this finite experiment. This does not imply calibrated safety under any unmodeled regimes.

## Experiment B: independent seed, 96 synthetic scenarios, equal 576 cell-query cap

[Independent-seed workflow and full JSON](https://github.com/aixaria0/strategic-intelligence-lattice/actions/runs/35794653292)

| Policy | Samples per context | Feasible-oracle agreement | Mean true constrained regret | Mean measured total wall time |
|---|---:|---:|---:|---:|
| Fixed full grid — larger-budget ceiling | 1152 | 49/96 | 0.07096 | 0.00278 s |
| Uniform — same budget | 576 | 29/96 | 0.10865 | 0.00253 s |
| Legacy ACA — same budget | 576 | 17/96 | 0.13569 | 0.00301 s |
| Single-batch C-VoC — same budget | 576 | 22/96 | 0.12366 | 0.00460 s |
| Multi-query predictive portfolio — same budget | 576 | 46/96 | 0.07573 | 0.01107 s |

For these 96 samples, the portfolio improved analytic feasible-action agreement and average regret over uniform at matched *cell-query count*. However, its Python planning overhead made mean wall time around 4.4 times uniform: **the experiment does not establish an advantage under equal measured CPU or wall-time budget**. Optimistically predetermined synthetic models are not real markets, military operations, or evidence of a new AI generalization theorem.

Family analysis on independent seeds (each 24 scenarios): crossing 8/24 vs uniform 7/24; near-safe 6/24 vs 0/24; uncertain boundary 10/24 vs 10/24; specially constructed certificate-ready 22/24 vs 12/24. The large average gain is concentrated in tasks where model-wide certification is the bottleneck. In 96 cases no returned *certified* action was analytically unsafe in the known toy. Exploratory actions remain uncertified.

## Experiment C: cache optimization and low-compute comparator

[Cached-forecast workflow and JSON](https://github.com/aixaria0/strategic-intelligence-lattice/actions/runs/35794779531) and [fast-target baseline workflow](https://github.com/aixaria0/strategic-intelligence-lattice/actions/runs/35794937028).

Deterministic memoization and exact impossible/certain-outcome short-circuits preserved the 46/96 portfolio agreements and 0.07573 mean regret. In the cached independent-seed run, median per-scenario wall time was 0.01047 s for portfolio versus 0.00273 s uniform, still substantially slower.

At equal 576 cell queries in the extended comparator run (96 cases): uniform 29/96 oracle agreements, predictive portfolio 46/96, cheap targeted method 32/96. The cheap method still had higher measured runtime than uniform on this Python implementation. No claim that heuristic replacement has reduced total computation is justified.

These are finite-seed, tiny-millisecond timing measurements on shared GitHub runners: use independent machines, repeated CPU measurements and larger problem distributions before making system-wide performance claims. Model uncertainty and reward/safety dependence remain untested in this checkpoint.

## Engineering conclusion

The project has now demonstrated an actual task-specific *decision-quality* gain from planning a multi-model portfolio of queries rather than assigning samples uniformly. It has NOT demonstrated end-to-end compute efficiency or general superiority. Next falsifiable requirements are calibrated multi-batch cost-aware stopping, a budget curve at equal measured CPU, posterior misspecification and correlated reward/safety stress, and any low-overhead analytic targeting alternative that survives comparison with uniform. Retain every negative result alongside the positive certificate-ready results.
