# Architecture: actual implementation versus historical research proposals

## Runnable synthetic baseline
Human parameters -> two-dimensional state -> deterministic seeded random shocks shared between candidate actions -> 3 opponent regime hypotheses -> vectorized rollout batches -> fixed reward and risk/entropy diagnostics -> per-model empirical feasibility gate -> chosen synthetic action -> recorded next-state observation -> approximate Bayesian regime update -> independent matched no-action baseline -> CLI/optional Streamlit results -> optional nonacting OpenRouter interpretation.

Main modules: src/sil/core.py (NumPy dynamics/MC/binned terminal entropy), inference.py (log-space Bayesian posterior), engine.py (bounded toy tournament and heuristic adaptation), cli.py (terminal interaction), reframer.py (optional summary-only LLM), app.py (Streamlit terminal UI), tests/, scripts/benchmark.py.

## Preserved alternatives
Original S=[r,a,i,t,u], S=[P_o,P_a,I,R,U], S=[market_share,competitor_share,brand_strength,liquidity,volatility]. Conceptual six-layer architecture: isolated Core Entity LLM, parallel classical "Quantum Automata", Collapse Engine, Strategist Interface, Void Shield minimal memory, human constraint injector. Current model does NOT claim full implementation of these broad semantic abstractions.

Proposed future interfaces: domain adapters for research/market/decision support/geopolitical abstract scenarios; budget-aware search (UCT/MCTS, progressive widening, Thompson sampling/CEM), sparse trajectory DAG, prioritization by expected information gain per CPU cost, batch multiprocessing and optional Redis/Ray/Dask schedulers; retrospective empirical 2D plots/video; typed reproducible experiment registry. No such distributed service is deployed by this repository today.

Historical flaws retained in archive/: single-path variance mislabeled entropy, beta*same−alpha*same cancellation, fabricated "value" by max random score, unsupported Lyapunov and inevitability claims, constant 3D score planes called entropy basins, mock LLM endpoint called a genuine reframer. These are not copied into the tested code.

## Engineering invariants
Fixed seed and config -> same synthetic round record. Positive normalized opponent weights. Recorded true synthetic observation only, no arbitrary mock beliefs. Policy-selection weights cannot redefine the fixed reward used to judge gain. LLM cannot select or execute actions. Every result states its synthetic provenance. No general attractor or market predictive claims follow from toy simulations.
