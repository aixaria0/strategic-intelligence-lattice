# ACA v0.3 — query-level research laboratory

This work is **synthetic decision-science research**, not a deployment-ready market or geopolitical predictor. The v0.2 engine and all original claims/legacy sources remain untouched. Query lab is a separate falsifiable testbed at src/sil/query_lab.py.

## Why a second environment?

The previous additive simulator usually made one action obviously best: agreement with a holdout evaluator was 100% for both fixed and adaptive allocation. That cannot establish the value of choosing where to simulate. The new experiment samples **one (opponent hypothesis, action) cell** per query, rather than all actions in the selected model. Expected reward curves cross as contextual demand and stress change, and each non-baseline action has a model-dependent probability of violating a synthetic safety constraint. We can thus measure incorrect choices, wasted samples and false safety approval.

Each scenario is drawn deterministically from its seed, with three hypothetical regimes and actions. A hidden synthetic regime generates one independent observable evidence value; Bayes' rule converts prior weights to posterior weights **before** rollout experiments. Simulating hypothetical regimes does not provide fabricated real observations for belief updates. Per-cell rewards and safety outcomes are Bernoulli with declared probabilities; values in [0,1]. Action 0 is safe by construction of this *toy*, not by any universal theorem.

## What is being optimized and how it is checked

For action a, oracle conditional value V(a)=sum_m w_m p_reward(m,a). Oracle feasibility is min_m p_safe(m,a)>=theta, with action 0 guaranteed feasible. The oracle is analytic from synthetic model parameters and is NOT estimated from the samples used to select actions. Observed decision regret is max(0,V(a_oracle)-V(a_selected)). In addition, the code reports an **exploratory-only** action using empirical safety rates rather than confidence-certified safety. Its empirical apparent feasibility is NOT an authorization to execute; count the number of objectively unsafe exploratory signals separately from oracle agreement. Also report whether chosen action is truly feasible under every declared model: a lower-reward but unsafe choice is not excused by good apparent score.

Six comparison policies access identical cell-wise random streams: fixed (exhaustive 9*Nmax samples, **not** equal budget); uniform (same smaller budget); random (same smaller budget); adaptive (same smaller budget, may stop early); hybrid (50% uniform, then adaptive, same smaller budget); and reward-only EVSI (same smaller budget, exact in an independent Beta-Bernoulli reward model but not safety-aware). Always distinguish full-grid fixed ceiling from equal-budget comparators; saving samples relative to full-grid fixed alone cannot demonstrate optimal adaptive allocation.

The heuristic ACA query priority considers posterior probability, estimated reward dispersion, empirical action-value proximity and synthetic safety-threshold proximity, with an exploration floor. A separate hybrid policy allocates half its budget evenly before switching to this priority. Cheap intermediate moments are updated every batch; expensive KL bounds are evaluated only at scheduled checkpoints and for the final decision. This proxy is *not* an exact value-of-information computation, calibrated posterior predictive entropy or MCTS.

## Time-uniform conservative safety and stopping

Under **independent identically distributed Bernoulli outcomes within every fixed cell**, and a posterior fixed before simulated data, the implementation now inverts the binary Kullback–Leibler divergence instead of using a symmetric Hoeffding radius. For every cell and every potential sample size n up to Nmax, set:

\[
c=\log(4K N_\mathrm{max}/\delta),\qquad
I_n(\hat p)=\{q\in[0,1]:n D_\mathrm{Bern}(\hat p\Vert q)\le c\}.
\]

The conservative lower/upper endpoints of I_n are used for BOTH stochastic reward and chance-constraint probability. A Chernoff tail bound followed by a union bound over two sides, both channels, all K=9 cells and all n<=Nmax gives familywise failure probability <=delta under the specified iid model. The fixed posterior is formed from one independent observation BEFORE simulation. Checkpoints use the same time-uniform bounds, so looking at partial results does not invalidate the stated model-conditional guarantee. The binary-KL interval is narrower for probabilities near 0 or 1, but can still require large samples when true safety is close to its required threshold. It does **not** certify the probability model itself, latent regime coverage, the real world or behavior under reward-distribution drift. The code certifies an action as safe only when EVERY model's lower confidence bound exceeds theta; if none qualifies it selects the toy's known-safe no-op.

An optional computational stop requires that the selected action be certified safe and its weighted lower reward bound exceed the weighted upper bound of every other action *not yet certified unsafe*. On the joint confidence event and under the specified conditional oracle and model, a resulting early-stop action is optimal among feasible actions. If the event fails, or priors/safety models are wrong, the claim does not apply. Otherwise the algorithm spends its budget and returns a conservatively safe action without a best-arm certificate. Reporting the certificate flag separately prevents confusing safe fallback with certified optimality.

## Reproduce

    python -m pip install -e '.[test]'
    python -m pytest -q
    python experiments/query_budget_benchmark.py --seeds 48 --budget 720 --maximum 256 --output query-results.json

The JSON contains complete per-scenario outcomes, model posterior, action, analytic oracle, true safety, true regret, samples per cell, stopping reason and measured wall time. Compare scenario diversity (number of distinct oracle actions) and report all tested action counts, not just average regret. Benchmarks on GitHub Actions should vary sample budgets, seeds and model assumptions before any efficiency claims.

## Open scientific limitations

Even binary-KL time-uniform concentration can be overconservative and may always select no-op under a small budget. Query priority may overfocus on near-threshold safety and neglect crossing expected-payoff signals. Fixed-world Bernoulli probabilities are simpler than real sequential strategic games; they are a contextual best-arm and safety identification task, **not** MCTS, true adversarial reasoning or proof that entropy controls the world. Improvements require multiple scenario distributions, decision error analyses, calibrated finite-sample risk confidence, CPU-matched budgets, model misspecification stress tests and ablation of priority terms.
