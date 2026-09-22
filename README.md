# Strategic Intelligence Lattice (SIL)

A reproducible, human-directed **synthetic** multi-agent research simulation platform with **Adaptive Computation Allocation (ACA)**, plus a preservation archive of **every major concept, rejected prototype and unproved mathematical claim** from the design discussion.

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

## Scientific status and honesty

Original "Quantum Automata" refers to parallel **classical** hypothetical models, not quantum hardware. A low-entropy outcome distribution does not imply high probability of favorable outcomes or inevitable victory. Old 3D constant planes are not measured entropy basins. The historical Flask "Core LLM" was a string formatter rather than an actual language model. Describing these early ideas as operational or publication-proven would be incorrect.

**Archival completeness:** The GitHub archive provides a detailed reconstructed chronology, original paper, selected original and reconstructed code, and a formal-claims registry. It is **not yet a byte-for-byte export of every source-chat message and every code block**; that requires an authorized raw conversation export. Track lossless parity in issue #1. No earlier idea should be erased merely because it was superseded.

## Security and provenance

This repository is PUBLIC. Never commit API keys, private credentials, unredacted personal chat exports, or proprietary data. Keep seed, parameters, simulator version, baseline, independent holdout evaluations and true failure results with every experimental assertion.
