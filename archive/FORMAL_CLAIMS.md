# Historical mathematics: original claims, statuses, counterexamples

**Preservation without endorsement.** These claims were discussed, including mutually incompatible variants. Some were presented prematurely as theorems; none become true simply because recorded here.

## State and feasibility [DEFINITION]
S=[r,a,i,t,u] resources, opponent capability, information, time, uncertainty; later S=[P_o,P_a,I,R,U] and S=[market_share,competitor_share,brand_strength,liquidity,volatility]. An initial toy used three variables; current synthetic implementation uses a two-dimensional unit-interval state. Do not mix state schema versions.
S_(t+1)=f(S_t,A_t,A_a,t,epsilon_t); toy f=S+action−opponent_action+noise. Actions A={alpha_i} map states to states with cost g(S,alpha_i). Feasible Omega={S:C(S)<=0}. Drafts conflated a linear constraint matrix and nonlinear constraint functions.

## Opponent hypothesis ensemble [DEFINITION]
O_a in {O_a^1,...,O_a^k}; opponent chooses an action to optimize a hypothesized objective. Beliefs about hidden motives are models, not established facts.

## Pressure [INVALID AS SCALAR SCORE]
P(S)=grad[V(S)−lambda*O_a(S)] then P(S)=grad[−alpha*H+ +beta*H−]. Gradient is a vector, not a scalar utility, selection score, favorable-hitting probability or guarantee of motion. The alleged integral integral P(S_t)dt was not a specified scalar line integral, and no transition-law relation was established.

## Entropy regions [HYPOTHESIS / NEEDS DEFINITION]
Omega+={S:V(S)>0}; Omega−={S:V(S)<0}.
H+=H(S_(t+1) | S_t in Omega+); H−=H(S_(t+1) | S_t in Omega−).
Proposed V_entropy=−alpha*H+ +beta*H− +gamma*L(S) (+delta information asymmetry in one variant), sometimes with alpha>beta>gamma>delta.
Such conditional entropies depend on the kernel AND the occupancy distribution of conditioned states; they are not automatically functions of a single S. On continuous state spaces differential entropy needs explicit measure and estimation choices.

## Inevitability [INVALID AS GENERAL THEOREM]
Original: reducing favorable transition entropy while increasing opponent-region entropy yields stochastic drift to a favorable attractor and "inevitability in expectation". Counterexample: a two-state chain + and − with P(+→−)=1 and P(−→−)=1 has zero transition entropy but ends at −. Altering entropy alone does not force favorable reachability, persistence, or hitting probabilities. No general convergence/novelty guarantee has been established.

## Variance [INVALID AS GENERAL ENTROPY ESTIMATOR]
Previous snippets substituted variance of a single path or of pooled time-step states for conditional entropy. Equal variance does not imply equal entropy. For a scalar Gaussian only, h(X)=1/2 log(2*pi*e*sigma^2). Pooling across times folds drift into dispersion. Current v0.1 finite-bin terminal Shannon entropy is a separate explicit empirical statistic.

## Stability and Lyapunov [UNPROVEN FOR PROJECT]
Proposed L(S)=||S−S*||^2, E[L(S_(t+1))−L(S_t)]<=0 and stronger E[Delta L]<=−c||grad V||^2; earlier text asserted that bounded noise + high alpha entropy penalty would yield this and make Omega+ invariant. No valid derivation was supplied. Counterexample: s_(t+1)=s_t+epsilon_t near s*=s_t, with zero-mean nonzero-variance noise; E[Delta L]=E[||epsilon_t||^2]>0, irrespective of a score penalty.
Valid conditional lemma: IF E[L(S_(t+1))|S_t]<=rho L(S_t)+b for all states, 0<=rho<1, b>=0, THEN E[L(S_t)]<=rho^t L(S_0)+b(1−rho^t)/(1−rho). This bounds expected L but does not establish its premise for this algorithm or almost-sure convergence.

## Bayesian adaptation [CONDITIONAL IDENTITY]
w_i(t+1)=ell_i(obs_t) w_i(t) / sum_j ell_j(obs_t) w_j(t). Earlier exp(−norm(error)) was called a Gaussian proxy; it is not a declared Gaussian likelihood. Updating weights cannot discover a missing opponent model. Near clipping boundaries, an untruncated Gaussian likelihood is only an approximation.

## Value hunting, regret, evolution [UNTESTED/INCORRECT HISTORICAL METRICS]
Original V_hunt=sum_trajectories pressure(tau)*basin_control(tau); later code defined V_hunt=max entropy score or beta*same_entropy−alpha*same_entropy. At equal alpha and beta that last expression is 0 for every candidate: no decision signal. High uncertainty does NOT establish high opportunity. Later independent realized utility minus matched no-action baseline is a measurable local toy quantity, not a general value-hunting guarantee.
"InevitabilityRatio=ExpectedGain/EntropyRisk" had no calibrated unit/denominator. Proposed regret sum O_a^*(S_t)−O_a(S_t) conflated comparator rewards and did not prove a regret bound. Alpha/beta sign-based self-evolution was an unvalidated heuristic; no adaptive improvement theorem.

## Experimental falsification [PROPOSED]
Compare against baseline no-op, random feasible, risk-aware without entropy, oracle synthetic opponent model, budget-matched fixed vs adaptive MC. Use paired random shocks, independent seeds, posterior calibration, confidence intervals, safety breach rates, CPU/RAM and latency. Never report empty results placeholders as observations.
