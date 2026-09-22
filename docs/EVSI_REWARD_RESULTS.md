# Reward-only EVSI experimental results

Scope: independent *synthetic* 24-scenario reward/safety experiments, not trading or real-world strategic validation.

GitHub Actions source, detailed JSON artifacts and exact code SHA: https://github.com/aixaria0/strategic-intelligence-lattice/actions/runs/35791622520

All equal-budget methods use the same model/scenario seeds and potential outcome cell streams. Full-grid fixed consumes 2304 cell queries; the other policies receive 720 or 1152 queries. All policies use the SAME time-uniform Bernoulli KL safety gate on the selected action. The exploratory signal does not receive safety certification.

## 720 queries

| Policy | Certified choice equals oracle | Exploratory-only choice equals oracle | Unsafe exploratory signals | Average elapsed time |
|---|---:|---:|---:|---:|
| Uniform | 6/24 | 18/24 | 4/24 | 0.00625 s |
| ACA heuristic | 5/24 | 17/24 | 3/24 | 0.01204 s |
| Exact reward EVSI | 6/24 | 16/24 | 5/24 | 0.02119 s |
| Fixed full grid (2304) | 7/24 | 21/24 | 1/24 | 0.00739 s |

## 1152 queries

| Policy | Certified choice equals oracle | Exploratory-only choice equals oracle | Unsafe exploratory signals | Average elapsed time |
|---|---:|---:|---:|---:|
| Uniform | 6/24 | 20/24 | 2/24 | 0.00610 s |
| ACA heuristic | 5/24 | 19/24 | 2/24 | 0.01213 s |
| Exact reward EVSI | 6/24 | 19/24 | 2/24 | 0.03110 s |
| Fixed full grid (2304) | 7/24 | 21/24 | 1/24 | 0.00750 s |

The one-batch reward EVSI is mathematically exact **within its stated independent Beta-Bernoulli model**, but it does not measure information about whether an action will become safety-certified. In these contexts, exact reward EVSI did not improve certified oracle agreement beyond uniform allocation and imposed much larger wall-time overhead. A good information-theoretic calculation can optimize the WRONG information objective. Accordingly, uniform querying is the default in the isolated query-lab UI/CLI; ACA/EVSI stay available explicitly for experimentation. The older multi-agent simulator retains its original default and historical findings.

The interpretation is descriptive for 24 cases; wall times are noisy single-run GitHub runner measurements. Even reward-only EVSI cannot be treated as a universal decision-improving strategy once safety constraints, model misfit or query costs change. The next objective for research is safety-aware, action-value-weighted *expected reduction of final constrained decision loss per actual CPU cost*, evaluated on independently seeded heterogeneous scenarios and compared to the simple uniform baseline.
