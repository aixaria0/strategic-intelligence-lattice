# SIL v0.5 — Constrained Value of Computation (C-VoC)

**Research status:** The code executes an explicit one-batch, joint reward-and-safety predictive decision-value calculation under a narrow independent Beta–Bernoulli toy model. It does not solve unconstrained strategic planning, robust real-world market decisions, an MCTS tree, or globally optimal compute allocation. The baseline policy remains **uniform** until repeated out-of-sample evidence proves otherwise.

## Measurable decision question

The legacy reward-only EVSI prioritized reward information, even when the true barrier was insufficient evidence to certify an otherwise profitable action's safety. C-VoC instead evaluates possible one-batch outcomes for BOTH channels of a (model, action) query, and recomputes the possible **certified-action set** before evaluating decision reward.

For each cell (m,a), reward success probability R_ma and safety success probability Q_ma have separate independent Beta(1,1) priors. After n independent Bernoulli simulated trials with r rewards and s safe outcomes, posterior R_ma~Beta(1+r,1+n−r) and Q_ma~Beta(1+s,1+n−s). Under the model, the independent predictive number of reward and safety successes from b new paired simulated trials are two Beta–Binomial(b,alpha,beta) draws. Dependence between safety and reward **would invalidate factorization and require a joint model**.

Let C(D) be the finite set of candidate actions certified by the original time-uniform KL frequentist safety gate based on observed synthetic safety counts (plus toy no-op 0, assumed safe by construction). Let

\[
V(D)=\max_{a\in C(D)}\sum_m w_m\mathbb E[R_{ma}\mid D],
\]

where model weights w_m come from a distinct synthetic external observation and are held fixed while evaluating simulated queries. For query q=(m,a), enumerate possible future reward/safety success counts (k_r,k_s), probability factor P(k_r|D)P(k_s|D), and compute

\[
\Delta_{\rm one}(q)=
\sum_{k_r,k_s}P(k_r,k_s\mid D)\,V(D,k_r,k_s)-V(D).
\]

This is an exact one-batch posterior-predictive calculation **for the defined decision rule, prior and independent channels** (subject to numeric precision), not a globally optimal or infinite-horizon planner. It need not be nonnegative: a new observation can *revoke* an existing conservative confidence-based action certification. Thus it is NOT conventional nonnegative unconstrained sample information. The implemented allocator uses positive signed Δ_one/b as a decision-priority proxy; the result also reports the expected positive increase separately. Safety certification of final chosen action is performed by the existing frequentist rule, **never by Bayesian means alone**.

## Governor and cost

For the experimental c_voc mode, after pilot coverage the algorithm recomputes one-batch query priorities at regular checkpoints. A zero one-batch value falls back to uniform allocation: one batch might not be enough to reach a certification boundary even though several could. Therefore **zero one-batch value does not justify global stopping**.

The separate c_voc_governor mode requires an explicit positive compute_price in units of *expected synthetic reward per simulated query*. It can stop only after minimum cell coverage >=max(pilot,64) and estimated one-batch value/query <= the user-specified price. This is an opt-in MYOPIC experiment and can stop too early: the approximate expected gain is not a confidence lower bound, the price is a user-supplied conversion factor, and the method does not yet optimize actual CPU. Results record stop reason, expected gain per query, rollout count, and measured planning versus sample-collection wall time; benchmarking must compare **total measured elapsed wall time**, not merely simulation draw counts.

## Launch

    python -m pip install -e '.[test,ui]'
    python -m pytest -q
    sil query-lab --scenario 2 --trials 256 --budget 720 --method c_voc --output cvoc.json
    sil query-lab --scenario 2 --trials 256 --budget 720 --method c_voc_governor --compute-price 0.0001 --output governor.json
    python experiments/voc_benchmark.py --seeds 12 --maximum 128 --budget 576 --output c-voc-576.json
    streamlit run app.py

In the terminal-style UI, enter /query 2 720 c_voc (or /query 2 720 c_voc_governor 0.0001). This command **only runs synthetic research**; an uncertified exploratory suggestion remains visibly non-actionable. Do not compare raw run-time timing floats for byte-identical deterministic audit: the RNG-dependent results are reproducible but wall clock is not.

## Falsification protocol

Three prespecified synthetic scenario families in experiments/voc_benchmark.py: (a) crossed contextual reward curves, (b) high-but-uncertain safety with reward-competitive actions, (c) safety probabilities near the threshold where certification is expensive or practically unavailable. Same independent cell-wise random stream prefixes per scenario and method. Compare fixed full-grid as an explicitly **higher-budget accuracy reference**, while uniform, heuristic ACA, hybrid, reward-only EVSI, C-VoC and the governor each receive identical smaller caps. Record oracle agreement, analytic regret, unsafe certified and unsafe exploratory choices, planner CPU versus simulation CPU, actual query count, and governor stop rate. Predeclared seed families prevent cherry-picking one showcase case. Findings apply only to that toy family. No performance improvement claim is warranted unless shown by the resulting comparisons.

## Mathematical sanity checks

A constructed case in tests/test_voc.py has reward action 1 clearly dominant but one remaining uncertain safety cell prevents certification. One extra reward-only query cannot change the reward ranking (reward EVSI zero), while a joint reward+safety query can sometimes complete the certification and has strictly positive one-step constrained decision value. Other tests check conservative fallback, budget accounting, explicit governor activation and deterministic seeded outcomes. They do NOT prove improvement over uniform or robustness to unknown real-world distributions.

## Research work beyond v0.5

A tractable full compute governor would need an estimate of *multi-batch* constrained decision regret reduction per actual measured CPU cost, well-calibrated model-misspecification detection, conservative sequential stopping and comparison under equal time budgets. An actual sequential MCTS planner requires branching state dynamics and cannot be validated using this one-shot contextual Bernoulli environment. Preserve all negative results and rejected mathematical hypotheses in archive/.
