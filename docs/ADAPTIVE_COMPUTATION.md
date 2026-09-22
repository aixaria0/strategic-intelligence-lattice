# Adaptive Computation Allocation (ACA) — experimental v0.2

> **Status:** implemented heuristic for the existing *synthetic* three-model / three-action environment. No claim of provably optimal allocation, exact value of information, formally certified best action, general real-world predictiveness or faster wall-clock execution.

## Problem

Fixed allocation spends N trajectories on every (action, opponent-model) pair regardless of how near competing choices or risk boundaries are. ACA uses a small pilot in every model, then decides **which opponent model** should receive the next batch. Within a selected model, it simulates **every candidate action using the same random shocks**. This preserves paired cross-action comparison and makes action-rollout cost explicitly measurable.

The long-horizon research target is a query-specific expected value of sample information per unit compute cost:

\[
\mathrm{EVSI}(q)/c(q) =
\frac{\mathbb{E}[\min_a \mathbb{E}(L\mid D)] -
      \mathbb{E}[\min_a \mathbb{E}(L\mid D,Y_q)]}
     {c(q)}.
\]

Here q is a proposed *simulation query*, D is current observations, and Y_q is its possible new result. The expectation is over a **specified** predictive distribution, and the numerator is nonnegative only for exact Bayesian decisions with free option to ignore new data under the specified model. Our v0.2 implementation does **not** compute EVSI; instead it uses a cheap model-weighted decision-gap / uncertainty / constraint-proximity proxy. This is deliberate: exact nested Monte Carlo VOI could cost more than the decisions it informs.

## Algorithm implemented

1. Human fixes the initial state, actions, reward function, minimum empirical safety rate, seed, max per-model samples, pilot, batch size and total action-rollout budget.
2. Pilot with 16 shock trajectories **per model** and **all actions** for each shock (3 models × 3 actions × 16 = 144 action-rollouts).
3. For each model/action calculate sample mean fixed reward, downside shortfall, conditional terminal histogram entropy if ≥12 favorable outcomes (otherwise conservative penalty 1), favorable-state occupancy and empirical path reserve-safety frequency.
4. Combine models using the current recorded-observation Bayesian posterior and the existing weighted scoring rule. Use the same empirical feasibility threshold as fixed allocation.
5. Prioritize the next model by a heuristic of posterior weight × top-two-action utility standard-error proxy plus a risk-boundary proximity term, adjusted for sample size. Sample all actions with shared shocks for that model. Unallocated models retain their own deterministic random streams.
6. Stop at human action-rollout budget or configured per-model maximum. Optionally stop when a large empirical decision gap exceeds a conservative *heuristic* uncertainty allowance and constraints appear clearly away from the threshold. This is not a theorem or formal risk certificate.
7. Record per-model counts, selected action, cumulative synthetic realized reward versus matched action-zero baseline, posterior update and stop reason.

The historical fixed-mode API and score function remain unchanged; ACA is opt-in. The simulator never executes an external action. Nothing in this code claims a genuine physical "entropy basin" or ensures positive real-world gains.

## Repeatability and experimental measurement

From repo root:

\`\`\`bash
python -m pip install -e '.[test,ui]'
python -m pytest -q
sil run --rounds 3 --trials 128 --allocation fixed --seed 2026 --output fixed.json
sil run --rounds 3 --trials 128 --allocation adaptive --budget 1152 --seed 2026 --output adaptive.json
python experiments/aca_benchmark.py --seeds 12 --trials 128 --holdout 512 --output aca-results.json
streamlit run app.py
\`\`\`

For 128 trials/model/action, fixed spends 3×3×128=1152 action-rollouts per agent per round; adaptive is *capped* at 1152 and may stop early. The pilot minimum is 144; a budget lower than the pilot is rejected. The cap is counted in simulated trajectories, not CPU time. Reanalysis of empirical distributions after each batch can outweigh savings: **always report wall time, memory and quality separately**.

The benchmark fixes an equal maximum budget, uses the same start state, prior and seed for fixed/adaptive evaluation, then scores selected actions on an independent held-out stochastic seed. Report selected-action agreement with held-out best feasible candidate, holdout feasibility, selection regret, actual rollout usage and measured wall time; include all raw records and run multiple independent seeds. A small exploratory benchmark does not establish general superiority.

## Failure modes / research tasks

The heuristic can allocate too many samples to an apparently influential model; a model posterior can be wrong or incomplete; low sample counts bias histogram entropy; quantile-based downside statistics have nontrivial finite-sample error; early-stop decision-gap allowance is **not a confidence bound**; empirical safety frequency can miss rare failures. Both simulated utility and feasible-region rules are design choices rather than evidence of external value. Next milestones: calibrated confidence sequences or empirical Bernstein safety bounds, paired-difference ranking, independent problem distributions and budget curves, amortized CPU profiling, then optionally VOI approximation and search-tree expansion (MCTS only once branching environments are available).

All earlier "inevitability" arguments remain preserved in archive/FORMAL_CLAIMS.md, not promoted to operational guarantees by this integration.
