# Exact one-batch Reward EVSI — research-only module

This module, src/sil/evsi.py, is a **precisely defined value of sample information calculation** in a small conjugate Beta–Bernoulli reward model. It is NOT an exact full strategic value-of-information planner or a substitute for chance-constraint certification.

## Assumptions

For each hypothesized regime m and candidate action a, an unknown reward success probability p_ma has an independent Beta(1,1) prior. We have observed s_ma successes in n_ma independent synthetic Bernoulli reward samples. The regime weights w_m come from the synthetic external evidence and are fixed during this computation; counterfactual simulation is not confused with a new real regime observation.

Posterior p_ma | D ~ Beta(1+s_ma,1+n_ma-s_ma). Its mean is μ_ma=(1+s_ma)/(2+n_ma). The estimated action value is V_a=Σ_m w_m μ_ma. A new query q=(m,a) returns a block Y of b Bernoulli reward outcomes. Under the posterior predictive model, K=sum(Y) follows Beta-Binomial(b,1+s_ma,1+n_ma-s_ma).

## Exact reward-only batch EVSI

Define the current maximum posterior expected reward V*=max_a V_a and V*_k the maximum posterior expected reward after hypothetical block count K=k. The module enumerates all k=0,...,b and computes

\[
 \mathrm{EVSI}^{\mathrm{reward}}_b(m,a)
 =\sum_{k=0}^{b}\Pr(K=k\mid D)\,V^*_k-V^*.
\]

For a fixed query, this is exact within the declared independent conjugate reward model up to floating-point arithmetic; no nested rollout is needed. Since E[μ'_{ma}|D]=μ_ma and max is a convex function, Jensen's inequality gives EVSI≥0 under this model. It may be exactly zero if one additional batch cannot change the posterior-best action; zero does NOT mean more observations have no value, especially if more than one batch is needed to cross a ranking boundary.

When a one-batch EVSI matrix is numerically zero across all remaining cells, the experimental allocator spreads samples uniformly rather than pretending a positive information gain. It uses an equal batch size per query so sorting EVSI is equivalent to sorting EVSI/query-cost except for the last partial batch, whose cost is still accounted for in the recorded rollout count.

## Critical safety limitation

The formula optimizes REWARD ONLY; it does not include a reward tradeoff against posterior safety risk, nor the possible future ability to certify the chance constraint. Consequently it is an **experimental ablation**, not the recommended live selector. The final action continues to require the independently computed time-uniform Bernoulli safety lower bounds. An exploratory action based on empirical safety rates is reported separately and never receives an execution authorization.

## Falsification

    python -m pytest -q tests/test_evsi.py
    sil query-lab --scenario 2 --trials 256 --budget 720 --method evsi_reward --output evsi-example.json
    python experiments/evsi_reward_benchmark.py --seeds 24 --maximum 256 --budget 720 --output evsi-report.json

The prespecified benchmark compares EVSI, uniform and legacy heuristic ACA at the same query budget and full-grid fixed at a larger accuracy ceiling. Report analytic-oracle agreement, decision regret, false-certification and unsafe exploratory counts, real elapsed time and query use. If EVSI cannot reduce decision loss or wall time under reasonable distributions, retain that negative result rather than claiming a new AI paradigm.

A future *full* safety-aware information planner would need a value function that incorporates chance constraints, a joint posterior for reward and safety, a model of available follow-up queries, and independently calibrated stopping rules. Exact reward EVSI is a tractable mathematical building block, not an assertion that such a planner is already present.
