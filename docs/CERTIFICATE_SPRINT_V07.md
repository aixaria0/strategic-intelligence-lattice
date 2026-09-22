# SIL v0.7 — Certificate Sprint and the bulk-uniform control

## A specific efficiency hypothesis, not a general AI theorem

The earlier finite Beta-Binomial multi-query portfolio improved constrained decision quality in a toy, but evaluating hundreds of hypothetical future outcomes consumed more CPU than actually simulating the chosen outcomes. Certificate Sprint makes a different tradeoff: after an all-cell pilot, it selects at most ONE promising non-baseline action using the fixed external model posterior, empirically measured reward, and pilot safety rates. It then gathers a large, committed block of reward and safety observations from ALL modeled regimes for that action. The final action still passes the original independent, time-uniform frequentist KL safety gate. The pilot selection is only a cheap heuristic; it does not claim probabilistic future certification, optimality, model coverage or real-world safety.

The control uniform_bulk also gathers each cell's planned samples in a few bulk simulator calls; it uses exactly the same pilot size, final KL gate, seeded per-cell/per-channel random streams, maximum per-cell samples and total cell-query budget as Sprint. For budgets divisible by the nine-cell grid, a unit test verifies uniform_bulk and the former small-batch uniform yield identical per-cell sample counts, realized reward/safety successes and selected actions. This is critical: Sprint is not credited with a supposed CPU advantage solely because uniform was inefficiently implemented.

## Schedule

For each of 9 model-action cells, collect pilot n0=16 independent synthetic Bernoulli reward and safety observations. Calculate Beta-smoothed expected reward for each action, weighted by model posterior derived from external synthetic evidence. Among actions whose pilot empirical safety rates are above the scenario threshold in EVERY modeled regime and whose estimated reward exceeds the safe synthetic no-op by >=0.018, choose the action with maximum positive-reward-advantage times product of Beta-smoothed safety rates. These 0.018 and threshold cutoffs are heuristic, not mathematically justified significance levels.

For the selected candidate, allocate as much of the remaining budget as affordable across all three model cells, with each cell capped at Nmax. Distribute any remaining budget evenly across the other cells. If the pilot suggests no candidate, evenly sample all cells. Generate ONLY sampled queries using independent seeded per-cell and per-channel streams, and retain all actual sampled success counts for audit. This schedule minimizes repeated Python planning and simulation-loop overhead; it does not do adaptive re-optimization after new data arrive.

The toy's action 0 is safe by definition of the synthetic simulator; this does not imply a real environment always offers a known-safe fallback. The separately reported exploratory action remains uncertified even if its empirical safety rates look high.

## Run

    python -m pip install -e '.[test]'
    python -m pytest -q tests/test_certificate_sprint.py tests/test_query_lab.py
    sil query-lab --scenario 2 --trials 128 --budget 576 --method certificate_sprint --output sprint.json
    sil query-lab --scenario 2 --trials 128 --budget 576 --method uniform_bulk --output uniform-bulk.json
    python experiments/certificate_portfolio_benchmark.py --seeds 24 --seed 202620 --maximum 128 --budget 576 --repeats 3 --output sprint-study.json
    python experiments/cpu_budget_frontier.py --calibration 8 --holdout 12 --maximum 128 --budget 576 --repeats 3 --output sprint-cpu-frontier.json

## Initial result and caution

The preliminary 96-context benchmark with seed 202610 and 576 queries recorded Sprint 45/96 feasible-oracle matches, mean true regret ~0.0784 and mean process CPU ~0.00272 s, versus *small-batch* uniform 31/96, regret ~0.1040 and CPU ~0.00295 s. A method-specific gain cannot be assigned from that comparison because both sampling policy AND batch handling differed.

A later separate 32-context CPU calibration at seed 202613 used uniform_bulk as its comparator, then evaluated fixed budgets on 48 disjoint contexts at seed 202614. Calibration selected 576 Sprint cell queries within bulk-uniform's mean CPU allowance. Holdout Sprint matched the analytic feasible oracle 22/48 times (regret 0.06982), against uniform_bulk 17/48 (regret 0.09179). **But** holdout measured Sprint mean CPU 0.00456 s and bulk-uniform 0.00440 s (~3.7% higher Sprint CPU). Calibration matching is therefore NOT a verified per-context or mean holdout CPU dominance claim; these are noisy millisecond measurements on shared GitHub runners.

Raw evidence: [CPU calibration and disjoint holdout](https://github.com/aixaria0/strategic-intelligence-lattice/actions/runs/35796378248) and [earlier Sprint benchmark](https://github.com/aixaria0/strategic-intelligence-lattice/actions/runs/35796218483).

## Decision rule for deployment

Uniform remains the default query-lab algorithm until Sprint is compared with uniform_bulk (not just small-batch uniform) across additional independent, unselected seeds, measured actual CPU, false safety certification, and divergent scenario models. A low fixed CPU budget must be calibrated BEFORE reading holdout outcomes, and any holdout CPU overrun must be reported. Even positive results are confined to synthetic fixed-distribution decision research; not real market prediction or autonomous strategic control.
