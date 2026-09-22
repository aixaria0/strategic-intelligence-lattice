# Catalog of historical code prototypes (archive index)

Original responses produced variants of code over many turns. The following are historical snippets and ideas, not asserted to be tested/deployed. This is an index, NOT a verbatim export of those source files.

| Prior filename or component | Historical content and limitation |
|---|---|
| LangChain OpenRouter Streamlit quickstart | ChatOpenAI sidebar/form with HumanMessage; initial demo. |
| state.py, dynamics.py, entropy.py, collapse.py, stability.py | State object, additive dynamics, pooled trajectory variance, highest-score action, norm stability gate; not general proofs. |
| agents/adversary_models.py | Two hardcoded opposing action vectors, not trained intent inference. |
| modes/research.py, decision_support.py, market_engine.py | Named architectural placeholders, not functional domain adapters. |
| llm/reframer.py | Old LangChain text prompt with no structured evidence validation. |
| ui/terminal_app.py | Text input and echo placeholder, not the integrated simulator. |
| rollout.py | Python-loop paths per trial and step; current implementation instead vectorizes trials. |
| AdversaryManager.update | Prior times exp(-prediction error); likelihood unspecified. |
| visualization.py | Tricontour per-state values labeled as entropy basin without a calibrated state-conditional estimator. |
| worker_node.py, /rollout, /bayes_update | Flask one-path worker and toy posterior endpoint. |
| coordinator.py, coordinator_dashboard.py | Sequential HTTP requests, max entropy selection, not a true distributed planner. |
| core_llm/core_reframer.py | Mock Flask service formats max pressure without any LLM request. |
| launch_demo.py, run_full_lab.py | Docker Compose launches and client-side tournament prints; no synchronized dashboard state or proofs of distributed execution. |
| coordinator_dashboard_full.py / heatmap.py | 2D metric plots + 3D constant planes. NOT empirical basin geometry. |
| coordinator_dashboard_video.py / dashboard_live_video.py | Kaleido/imageio MP4 capture; early demo data was randomly generated. |
| RGB surface and leaderboard variants | Plotly Surface facecolor not supported; 2D layout annotations do not accept z coordinate. |
| minimal_lattice_tournament.py / terminal_lattice.py / hybrid_lattice.py | CLI, on-demand plots/video; entropy_scores=np.random.rand was placeholder, not MC. |
| trajectory DAG, MCTS, Thompson Sampling, Cross-Entropy Method, priority queue | Proposed computation-efficient planner, not included in original runnable code. |
| Redis streams, Ray, Dask, Kubernetes, Apache Arrow, Polars | Architectural proposals, no deployed or benchmarked cluster. |

Original prototype equations, failure analysis and revisions appear in FORMAL_CLAIMS.md and DEVELOPMENT_TIMELINE.md. Preserve original snippets in archive/raw/ if a full conversation export becomes available; do not claim this summary is a lossless transcript.
