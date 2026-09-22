# Alternative architectures and strategic objectives

This document preserves alternatives, including ones not selected or supported.

**Objectives discussed:** (1) minimax stability, (2) dominance/asymmetry, (3) deception / opponent misestimation, (4) "inevitability" by entropy steering. Original favored inevitability with a stability floor and information-asymmetry modifier. This preference was a historical design choice, not an empirical finding that any of these objectives universally dominates.

**Six original layers:** isolated Core Entity (semantic synthesis; no raw data/tools); classical multiple-model "Quantum Automata" (not quantum computing); Collapse Engine (rank trajectories, not physical measurement); Strategist Interface (compressed futures and constraints); Void Shield (minimal structural memory); human constraint injector with final authority.

**Code and deployment variants:** pure Streamlit demo; single Python/NumPy simulator; agent objects versus contiguous arrays; Monte Carlo batch versus tree search; Flask HTTP workers plus coordinator and optional Redis/PostgreSQL; Docker Compose, multiprocessing, Ray/Dask/Kubernetes. Networked workers were proposed as a future scale option, not tested as a coherent concurrent implementation.

**High-value efficiency direction:** measurable utility under baseline comparisons; sequential allocation of simulation budget based on decision uncertainty; priority queue/value-of-information, UCT/MCTS/progressive widening when actions branch; bounded numeric state/posteriors rather than full narrative LLM memory; small checkpoint summary to optional LLM; text terminal first, graphics/video only on demand. Historical "5–20×" compute savings and "hundreds of agents in seconds" were not benchmarked and must not be reported as results.

**Domain candidates:** synthetic research lab; decision support; competitive financial markets; geopolitical/defense-themed *abstract simulation*; AI-vs-AI tournament. Only a synthetic toy research/tournament model was coded and tested in the v0.1 local package. Real-data adapters, financial performance and policy conclusions require distinct evidence.

**Provenance standard:** for every future improvement preserve original claim, current definition, model assumptions, reproducible seed, reward/constraints, comparator policy, measured outcome, limitations, and any discovered counterexample. Do not delete unsuccessful or speculative historical variants.
