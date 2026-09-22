# Historical failure conditions and counter-steering hypotheses

The original discussion explored potential weaknesses and abstract opponent responses. These are **research hypotheses**, not documented real-world outcomes or guarantees that favorable trajectories can be forced.

## Failure conditions

1. Model misspecification: wrong favorable/unfavorable partition stabilizes the wrong region.
2. Hidden state z: unobserved coupling invalidates the simplified transition model.
3. Opponent disturbance bandwidth: opposing feasible control set may overwhelm candidate steering actions.
4. Constraint over-tightening: too little flexibility may reduce shock tolerance.
5. Symmetric knowledge: both agents optimize known steering metric; unilateral fixed-adversary reasoning fails.
6. Switching opponent objective: stationary Bayesian model weights become stale; missing hypotheses cannot be recovered by reweighting alone.
7. Conditional entropy and occupancy: low entropy within favorable states says nothing by itself about the probability of entering them.

## Counter-steering ideas discussed

- Basin pollution: increase disturbances in currently favorable regions.
- False attractor: visible metrics lure selection toward a stable but low-utility region.
- Entropy equalization: competitor alters its own dispersion and action strategy.
- Objective obfuscation: assumed payoff or transition kernel stops matching the world model.
- Global uncertainty increase: weakens selective steering signal and calibration.

These generic abstract categories should be tested under explicit synthetic transition laws, held-out seeds and external fixed reward/constraint metrics. Neither increasing entropy nor drawing a visually deep basin establishes any strategic success.
