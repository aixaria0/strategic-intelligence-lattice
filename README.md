# Strategic Intelligence Lattice (SIL)

A reproducible, human-directed **synthetic** multi-agent research simulation platform with **Adaptive Computation Allocation (ACA)** and a separate, nontrivial query-level decision laboratory, plus a preservation archive of **every major concept, rejected prototype and unproved mathematical claim** from the design discussion.

## Start here

This repository deliberately separates the **current runnable baseline** from the **historical research archive**. The baseline is not a proven new AI paradigm, market prediction engine, real-world adversarial system, or formal stability certificate.

## Run

Python 3.10+:

    python -m pip install -e '.[test,ui]'
    python -m pytest -q
    sil run --rounds 10 --trials 128 --seed 2026 --output audit.json
    sil chat

ACA v0.2 (optional, fixed mode remains the baseline):

    sil run --rounds 3 --trials 128 --allocation adaptive --budget 1152 --seed 2026 --output adaptive.json
    python experiments/aca_benchmark.py --seeds 12 --trials 128 --holdout 512 --output aca-results.json

ACA samples all actions with shared shocks inside each selected opponent model, then allocates the next batch to the model with the largest heuristic decision-uncertainty/risk-boundary priority. See docs/ADAPTIVE_COMPUTATION.md. Sample-saving and wall-clock improvement are **empirical questions**, not guarantees.

Terminal commands: /help, /run 3, /agents, /inspect 1, /history, /reset, /quit. All actions are simulated. UI:

    streamlit run app.py

On Docker-equipped hosts:

    docker compose up --build

Open http://localhost:8501. The LLM is OFF by default; the optional /explain command uses OPENROUTER_API_KEY from environment variables, never a committed key. Docker Compose is a local single-service UI, **not** the previously sketched distributed worker cluster.

## What is in GitHub

- archive/DEVELOPMENT_TIMELINE.md — chronological design stages from the original OpenRouter quickstart through adaptive compute proposals.
- archive/FORMAL_CLAIMS.md — preserve early entropy, pressure, Bayesian, Lyapunov, inevitability, value hunting and regret assertions, including invalid and unproven ones, adjacent to counterexamples.
- archive/PROTOTYPE_CATALOG.md — inventory of earlier code variants, domains and UI proposals.
- archive/DESIGN_DECISIONS.md — objective-function alternatives, six cognitive layers, infrastructure variants and efficiency-first decisions.
- legacy/original_paper_draft.tex — original user-supplied manuscript including its historic claims and illustrative TikZ figures; not scientifically endorsed.
- legacy/snippets/ and legacy/variants/ — preserved historical *selected* code excerpts, including deliberately flawed prototypes.
- docs/ADAPTIVE_COMPUTATION.md — adaptive allocator, fixed-budget comparator, assumptions and benchmark protocol.
- experiments/aca_benchmark.py — independent-holdout exploratory comparison.
- docs/paper.tex — qualified working research note; docs/RESEARCH.md and ARCHITECTURE.md explain current scope and future falsification tests.
- src/sil/, app.py, tests/, .github/workflows/ci.yml — operational synthetic v0.1 engine, terminal, optional Streamlit, tests and CI.

## Query Lab v0.3 — equal-budget decision research

The new synthetic contextual lab includes crossing reward curves, external Bayesian evidence, an analytically evaluable feasible-action oracle, explicit chance constraints, cell-level adaptive sampling, and a strictly separate exploratory channel for uncertified opportunities. Its ten experimental allocation modes are fixed full-grid (cost ceiling), uniform, random, legacy adaptive, hybrid (half uniform / half adaptive), experimental reward-only EVSI, constrained C-VoC, opt-in C-VoC governor, predictive multi-query certificate_portfolio and plug-in certificate_targeted. It is NOT a market predictor or a general strategic control system.

    sil query-lab --scenario 2 --trials 256 --budget 720 --method hybrid --risk-threshold 0.75 --output query.json
    python experiments/query_budget_benchmark.py --seeds 48 --maximum 256 --budget 720 --output query-benchmark.json
    streamlit run app.py

In the terminal-style UI, type /query 2 720 hybrid after selecting 256 trials and suitable risk threshold in the sidebar. The certified action is the simulated admissible recommendation; the exploratory action is **research-only** and may be unsafe. Read docs/QUERY_LAB.md, docs/SAFETY_INFORMATION_LIMITS.md and docs/QUERY_LAB_RESULTS.md for assumptions, empirical results and failure modes.

## Evidence-based research defaults and reward EVSI

In the harder query-level testbed, uniform allocation currently outperforms the experimental ACA heuristic on conservative, model-conditional feasible action selection. The query-lab terminal and CLI therefore default to **uniform**. To study exact one-batch Beta-Binomial reward EVSI separately:

    sil query-lab --scenario 2 --trials 256 --budget 720 --method evsi_reward --output evsi-example.json
    python experiments/evsi_reward_benchmark.py --seeds 24 --budget 720 --maximum 256 --output evsi-results.json

The calculation in src/sil/evsi.py is exact for the declared independent reward-only Bayesian model, NOT the complete safety-constrained problem. It did not beat uniform on certified choice accuracy or wall time in the initial 24-scenario study. Read docs/EVSI_REWARD.md and docs/EVSI_REWARD_RESULTS.md for formulae, counterexamples and full linked experiments. See docs/QUERY_LAB_RESULTS.md for the 48-scenario lower/higher-budget comparison and docs/SAFETY_INFORMATION_LIMITS.md for the conditional sample-information bound.

## C-VoC v0.5 — Joint reward+safety value of computation

The new module computes a *one-batch* posterior predictive change in the best **safety-certified** expected reward after observing possible paired Bernoulli reward and safety outcomes, under explicitly independent Beta–Bernoulli priors. It adds cost-normalized optional query selection and an explicit price-based experimental governor. It never relaxes the final frequentist safety gate. C-VoC is not a solved multibatch planner or proven faster/better than uniform; the first 36-scenario test matched uniform on certified oracle agreement while consuming more wall time. Therefore uniform remains the query-lab default.

    sil query-lab --scenario 2 --trials 256 --budget 720 --method c_voc --output c-voc.json
    sil query-lab --scenario 2 --trials 256 --budget 720 --method c_voc_governor --compute-price 0.0001 --output governor.json
    python experiments/voc_benchmark.py --seeds 12 --maximum 128 --budget 576 --output c-voc-benchmark.json

The terminal-style UI supports /query 2 720 c_voc, /query 2 720 c_voc_governor 0.0001 and /query-report for complete results. Read docs/CONSTRAINED_VALUE_OF_COMPUTATION.md, docs/C_VOC_RESULTS.md and docs/paper.tex. The original historical archive and all earlier experimental modes remain present.

## Multi-Query Certificate Planning v0.6

Two experimental policies now plan a **set of future safety queries across all three modeled regimes**, rather than requiring each one-batch query to immediately change the authorized decision. Predictive certificate_portfolio enumerates finite Beta–Binomial cell-certification forecasts and ranks *precommitted* multi-model plans. certificate_targeted is a cheaper uncalibrated plug-in alternative; neither bypasses the original conservative KL safety gate. Exact multi-stage metareasoning and real-world safety are NOT implemented.

Crucial computational correction: Query Lab now uses **lazy, independent seeded generators for each model/action and each reward/safety channel**. Only selected cell queries actually produce Bernoulli observations. Earlier pre-lazy benchmarks remain archived with their original workflow commits; they are not directly bit-identical to current lazy-sampler experiments.

    sil query-lab --scenario 2 --trials 128 --budget 576 --method certificate_portfolio --output portfolio.json
    sil query-lab --scenario 2 --trials 128 --budget 576 --method certificate_targeted --output targeted.json
    python experiments/certificate_portfolio_benchmark.py --seeds 8 --maximum 128 --budget 576 --output portfolio-benchmark.json
    python experiments/cpu_budget_frontier.py --calibration 8 --holdout 12 --budget 576 --maximum 128 --repeats 3 --output cpu-frontier.json

**Measured result, not a general guarantee:** In 96 independent synthetic contexts, the predictive portfolio matched the analytically feasible oracle in 47/96 cases at the same 576 *actually generated* cell queries, against 31/96 for uniform. However, portfolio took ~3.4x the mean measured CPU. With a separately calibrated equal-CPU budget, the portfolio had to use only its 144-query pilot and matched the oracle in 4/48 holdout contexts, versus uniform's 12/48 at 576. Uniform therefore remains the default. Full context, raw workflow links, mathematical limits and explicit negative findings: docs/MULTI_QUERY_CERTIFICATE_PLANNING.md, docs/MULTI_QUERY_RESULTS.md, docs/CPU_BUDGET_RESULTS.md.

## Scientific status and honesty

Original "Quantum Automata" refers to parallel **classical** hypothetical models, not quantum hardware. A low-entropy outcome distribution does not imply high probability of favorable outcomes or inevitable victory. Old 3D constant planes are not measured entropy basins. The historical Flask "Core LLM" was a string formatter rather than an actual language model. Describing these early ideas as operational or publication-proven would be incorrect.

**Archival completeness:** The GitHub archive provides a detailed reconstructed chronology, original paper, selected original and reconstructed code, and a formal-claims registry. It is **not yet a byte-for-byte export of every source-chat message and every code block**; that requires an authorized raw conversation export. Track lossless parity in issue #1. No earlier idea should be erased merely because it was superseded.

## Security and provenance

This repository is PUBLIC. Never commit API keys, private credentials, unredacted personal chat exports, or proprietary data. Keep seed, parameters, simulator version, baseline, independent holdout evaluations and true failure results with every experimental assertion.
