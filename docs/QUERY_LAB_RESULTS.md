# ACA v0.3 query-level results — synthetic exploratory checkpoint

**Do not interpret this record as proof that adaptive allocation is superior.** These results were generated in GitHub Actions at https://github.com/aixaria0/strategic-intelligence-lattice/actions/runs/35790861921 . The run includes full JSON artifacts for all 48 independent contexts at each budget (including cell counts, analytic oracle, safety diagnostics and model posterior).

## Experimental setup

- 48 seeded synthetic scenarios with crossed action reward probabilities, three hypothetical regimes, three actions and model-dependent chance constraints (theta cycles over 0.72, 0.78, 0.84).
- The oracle is computed analytically from the declared synthetic reward/safety probabilities, rather than the finite Monte Carlo rollout set.
- Full-grid fixed samples all 9 cells to 256 draws = 2304. Uniform/random/adaptive/hybrid share the same LOW/MEDIUM/HIGH budget (288/720/1152), same potential-outcome streams and the same independent Bayesian evidence.
- The certified action requires time-uniform Bernoulli KL lower safety bounds for all declared models; exploratory-only actions rely on empirical safety frequencies and are **not permitted as operational recommendations**.
- Wall-clock timing is microsecond-scale Python/GitHub-runner noise and does not constitute a robust equal-CPU design without repetition and confidence intervals.

## Results and falsification

The analytical oracle selected action 0 in 11 contexts, action 1 in 30, and action 2 in 7. Thus, unlike the old additive benchmark, different actions can be optimal.

| 48 scenarios, high-budget comparison | Mean simulated cell draws | Certified action matches oracle | Exploratory-only action matches oracle | Truly unsafe exploratory signals |
|---|---:|---:|---:|---:|
| Full-grid fixed (2304) | 2304 | 19/48 | 40/48 | 2/48 |
| Uniform (1152) | 1152 | 13/48 | 40/48 | 3/48 |
| Random (1152) | 1152 | 11/48 | 38/48 | 4/48 |
| Pure ACA (1152) | 1152 | 12/48 | 37/48 | 3/48 |
| Hybrid (1152) | 1152 | 11/48 | 40/48 | 2/48 |

Certified choices were analytically safe in these sampled scenarios, but the conservative scheme fell back to action 0 for most of them. The ACA and hybrid heuristics did **not** outperform the uniform comparator in certified oracle agreement or exploratory oracle agreement. At the smallest and medium budgets, all 48 scenarios fell back to action 0 for uniform/random/adaptive/hybrid. This is useful negative evidence: allocating sample budget adaptively cannot compensate for too little statistical evidence to certify a high-safety alternative.

Exploratory-only suggestions identify additional opportunities but also make unsafe choices, so replacing the certified action with those suggestions would hide safety failures. The KL interval improves certification relative to a symmetric Hoeffding interval near high safety probabilities, but does not eliminate the information requirement.

## Scientific and engineering implications

1. Sample efficiency and wall-time efficiency remain separate questions; ACA uses more Python overhead than the uniform allocator at these small budgets.
2. The next experiment should vary the number of available safety samples and the distance between true safety p and theta, testing the information limit in docs/SAFETY_INFORMATION_LIMITS.md.
3. Include independent scenario distributions, possible model misspecification, candidate-specific reward-cost curves, familywise false-certification coverage, and equal CPU-budget comparisons. Retain uniform and no-op baselines so no advantage is asserted without a measurable gain.
4. Query-level allocation is a contextual **best-arm and constraint-identification testbed**, not yet MCTS or a real-market strategy.

Nothing in this report claims mathematical inevitability, a validated proprietary trading edge or a general-purpose strategic superintelligence.
