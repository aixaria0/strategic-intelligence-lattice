# SIL v0.6 — finite multi-query certificate portfolios

**Status:** implemented, synthetic, testable heuristic. This is NOT exact multistage Bellman planning, not a proof of global optimality, and not a real-world defense, market, or geopolitical decision engine. Earlier v0.5 C-VoC and uniform baseline are preserved intact. Uniform remains the default until independent CPU-matched results justify change.

## The problem one-batch C-VoC missed

Suppose action a has high posterior expected reward but no safety certification in THREE modeled regimes. With n=16 per cell and a severe time-uniform chance-constraint threshold, one new batch of 16 samples in just one model cannot change the final set of certified actions. One-batch EVSI for every individual safety query can be exactly zero, yet sampling another 32–64 draws in ALL three safety cells may jointly provide enough evidence to make action a admissible.

A stopped run must distinguish:
- One-step information gain zero: a single new batch does not change the final decision *now*.
- Multi-query information gain nonzero: a committed sequence of different queries could change it *after* enough evidence is gathered.
- A valid safety decision: regardless of forecasts, the observed-data frequentist KL confidence rule must still certify an action. A forecast alone never authorizes it.

## Algorithm

The new module src/sil/certificate_planner.py forms a finite, resource-limited portfolio targeted at one unapproved, high-posterior-reward action. For each of its three hypothetical opponent-model cells, it enumerates a coarse precommitted set of possible future draw counts: 0, batch, 2×batch, 4×batch, ... and the cell's remaining sample capacity. For a cell with n current Bernoulli safety samples and s successes, and b more draws, it computes

\[
P_{\mathrm{cert}}(n,s,b)
=\sum_{k=0}^{b}
P(K=k\mid s,n,b)\,
\mathbf 1\!\left\{
\frac{s+k}{n+b}>\theta,\;
(n+b)D_{\mathrm{Bern}}\!\left(\frac{s+k}{n+b}\middle\|\theta\right)
>\log\frac{4\,M\,N_{\max}}{\delta}
\right\},
\]

where M=9 model-action cells, K follows the Beta-Binomial posterior predictive count under the independent Beta(1,1) *planning* prior, theta is the declared safety threshold, and the cutoff is identical to the existing time-uniform KL screening policy. The exact probability formula is conditional on these specific modeling assumptions; the real frequentist safety gate does not depend on trusting the Bayesian predictive forecast.

For a fixed precommitted portfolio across three models, the assumed independent-cell forecast of completing an action certificate is the PRODUCT of the three cell probabilities. A future-certified action's present estimated reward advantage over the current certified incumbent is multiplied by this joint probability and divided by the total planned cell queries. This is an **expected-positive-opportunity heuristic**, not exact multi-step expected regret reduction. It freezes current reward posterior means, uses a coarse grid of plans, and does not optimize all conditional subsequent adaptations.

The numerical planner searches plans within the remaining query budget and per-cell maximum. If a plan exists, the simulator executes its separate model queries in batches using pre-generated, identical query streams for all compared policies; a new plan is considered after the committed portfolio is complete. Otherwise it continues fair uniform coverage for a bounded interval and retries; a missing plan must NOT be interpreted as global zero information value. The final action always passes the same original time-uniform KL confidence gate as the baselines. Simulated reward/safety channels are independent only in this declared toy.

## Reproducibility and experimental design

    python -m pip install -e '.[test]'
    python -m pytest -q tests/test_certificate_planner.py
    sil query-lab --scenario 2 --trials 128 --budget 576 --method certificate_portfolio --output portfolio.json
    python experiments/certificate_portfolio_benchmark.py --seeds 8 --maximum 128 --budget 576 --output portfolio-study.json

A dedicated GitHub workflow uses four families: crossing payoff curves, near-safety-boundary decisions, truly ambiguous near-threshold constraints, and a *purpose-built certificate-ready* family with one valuable and highly safe non-baseline action whose certification requires model-wide evidence. All four families and their random distributions are defined in the experiment source. The purpose-built family tests existence of a multi-query opportunity; it is not representative evidence about arbitrary real-world tasks.

The same potential-outcome draw prefix is used for fixed full-grid, uniform, legacy ACA, single-batch C-VoC and portfolio. Fixed costs 9×Nmax and is an explicitly higher-budget accuracy reference. Remaining methods have identical maximum query budgets, while individual methods may stop early if they meet their stated conditions. Report independent oracle choice, true constrained decision regret, analytically unsafe certified and exploratory outputs, query draws, planning overhead, **total measured wall time**, and whether any useful portfolio was actually constructed. A second independent-seed holdout uses 96 contexts and repeats each method's wall-clock timing three times; results must not be cherry-picked from the purpose-built family.

## Risk and efficiency limits

- Forecast independence across reward and safety cells can fail in realistic environments.
- A 1.5% minimum forecast threshold and cost-normalized portfolio ranking are explicitly heuristic engineering choices, not optimized or confidence-calibrated limits.
- The model posterior is presumed correctly obtained from separate evidence and held fixed during these counterfactual simulator queries.
- A candidate may be actually unsafe despite optimistic predictive forecast; observed-data safety certification is still mandatory.
- The planner does additional CPU work (Beta-Binomial predictive sums and finite portfolio enumeration). Fewer rollout calls alone are not evidence of CPU/energy savings.
- A method targeting only an already-estimated reward advantage may miss reward uncertainty or a safe opportunity not yet identified; simple uniform sampling often remains competitive.
- An absence of a plan in a finite coarse search is NOT evidence that all multistep plans are worthless.

## Research falsification priorities

Use independent held-out context seeds and more diverse distributions; vary budget and time-uniform confidence costs; test misspecified priors, dependent reward/safety draws and abrupt regime changes; compare at equal *wall/CPU budgets*. Improve caching/compiled predictive probabilities only if a profiler identifies bottlenecks. A globally optimal finite-horizon meta-MDP would require belief-dependent branching and Bellman evaluation over all possible future observations; this v0.6 portfolio is a deliberately smaller, auditable approximation.
