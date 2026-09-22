# Research integrity and testable scope

SIL v0.1 is an illustrative **synthetic** environment, not a validated new intelligence paradigm, market edge, geopolitical predictor or convergence theorem. Entropy is a property of a probability distribution, not a force. Empirical safety frequency is not an invariant. Keep ALL original disproven and untested claims in archive/FORMAL_CLAIMS.md rather than deleting them.

## Fixed toy definitions

State S=(value,reserve) in [0,1]^2. Action a in {0,0.5,1}; opponent hypothesis m in {0,0.5,1}. Clipped stochastic transition from src/sil/core.py with independent Gaussian perturbations. Favorable event F={value>=0.60 and reserve>=0.20}. Estimate p(F) as terminal occupancy; favorable entropy as normalized plug-in Shannon entropy from 5x5 terminal bins, conditional on F, with at least 12 favorable samples. For lower counts, the code uses a conservative selection penalty (this is not an entropy estimator).

Fixed utility R(S,a)=value+0.35*reserve-0.04*a. Selection score = expected R - alpha*(downside shortfall) - 0.1*beta*(favorable terminal entropy). Empirical feasibility gate checks probability of reserve remaining >=0.15 under EACH tested opponent model. The fallback action 0 can still be unsafe; the code labels it explicitly.

Posterior over three hypotheses updates from the actual synthetic next-state observation, with 15% prior mass refresh allowing regime switching. Gaussian likelihood is approximate near clipping boundaries. Posterior confidence cannot reveal missing hypotheses. Every counterfactual action uses the same random realization for paired value comparisons.

## Experiments required before research claims

Compare action-zero baseline, random-feasible, oracle-known synthetic model, no-entropy ablation, no-downside ablation, fixed-budget MC versus adaptive computation. Hold out seeds independent from model selection; report confidence intervals, sample count, favorable occupancy, violations, posterior calibration, external reward uplift, simulator calls, CPU, RAM and target machine. Predeclare stopping rules; never promote illustrative charts into evidence of physical attractors.

## Mathematical correction to early manuscript

Neither reducing H+ nor increasing H− guarantees p(F), reachability or favorable invariance. Lyapunov descent requires an independently verified drift or barrier inequality for the specified transition kernel; bounded perturbation and a chosen score alone do not prove it. docs/paper.tex states a conditional bound only, while archive/FORMAL_CLAIMS.md retains the stronger original invalid claims and counterexamples.

## Domain adapters still proposed

Market research needs timestamped ingestion, fees, slippage, splits, market hours, leakage-free walk-forward validation and external baselines. Defense/geopolitical research is an abstract scenario modeling proposal, not an assessment of any real actor. Distributed schedulers, trajectory DAG, MCTS and value-of-information priority sampling require implementation and measurable resource-budget comparisons.
