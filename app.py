"""Lightweight terminal-style Streamlit human control and on-demand charts."""
from __future__ import annotations
import json
import streamlit as st
from sil.core import Config
from sil.engine import Tournament
from sil.reframer import explain
from sil.query_lab import QueryConfig, make_scenario, allocate

st.set_page_config(page_title="SIL / Research Terminal", layout="wide")
st.markdown("""<style>
.stApp {background: #10151b; color: #d5e3dc;}
[data-testid="stChatMessage"] {border: 1px solid #283944; border-radius: 5px;}
code, pre {color: #a6e0c8 !important;}
</style>""", unsafe_allow_html=True)
st.title("SIL / Strategic Intelligence Lattice")
st.caption("SYNTHETIC RESEARCH ONLY · No market feed, adversarial deployment, or action execution")
with st.sidebar:
    st.header("Human constraints")
    seed = st.number_input("Seed", min_value=0, max_value=2**31-1, value=2026)
    trials = st.select_slider("Monte Carlo trials / model", options=[16, 32, 64, 128, 256, 512], value=128)
    horizon = st.slider("Rollout horizon", 1, 32, 8)
    n_agents = st.slider("Research agents", 2, 12, 3)
    safety = st.slider("Minimum empirical safety / model", 0.0, 1.0, 0.80, 0.05)
    evolve = st.toggle("Heuristic parameter exploration", value=True)
    allocation_mode = st.selectbox("Computation allocation", ("fixed", "adaptive"))
    full_cap = 9 * trials
    allowed_budgets = [value for value in (144, 288, 576, 1152, 2304, 4608)
                       if value <= full_cap]
    budget = st.select_slider("Action-rollout budget / agent / round",
                              options=allowed_budgets, value=full_cap,
                              disabled=allocation_mode == "fixed")
    st.caption("Adaptive: shared shocks across actions within each model. Budget counts action-rollouts, not wall time.")
    show_chart = st.toggle("On-demand reward chart", value=False)
    if st.button("Reset lab"):
        st.session_state.pop("lab", None)
        st.session_state.pop("transcript", None)
        st.rerun()

config = Config(trials=trials, horizon=horizon, seed=int(seed), min_safe_probability=safety)
signature = (seed, trials, horizon, n_agents, safety, evolve, allocation_mode, budget)
if st.session_state.get("signature") != signature or "lab" not in st.session_state:
    st.session_state.lab = Tournament(config, num_agents=n_agents, evolve=evolve,
                                      allocation_mode=allocation_mode,
                                      simulation_budget=budget if allocation_mode == "adaptive" else None)
    st.session_state.signature = signature
    st.session_state.transcript = [{"role": "assistant", "content":
        "SIL ready. /run [n] · /query [scenario] [budget] [method] · /agents · /inspect N · /history · /explain · /help"}]
lab = st.session_state.lab
for message in st.session_state.transcript[-18:]:
    with st.chat_message(message["role"]):
        st.code(message["content"])
command = st.chat_input("sil> /run 3")
if command:
    st.session_state.transcript.append({"role": "user", "content": command})
    parts = command.strip().split()
    verb = parts[0].lower().lstrip("/") if parts else ""
    try:
        if verb == "run":
            count = int(parts[1]) if len(parts) > 1 else 1
            lab.run(count)
            output = f"Completed round {lab.round}.\n" + json.dumps(lab.leaderboard(), indent=2)
        elif verb == "query":
            scenario_id = int(parts[1]) if len(parts) > 1 else 0
            query_budget = int(parts[2]) if len(parts) > 2 else 9 * trials
            query_method = parts[3].lower() if len(parts) > 3 else "uniform_bulk"
            query_price = float(parts[4]) if len(parts) > 4 else 0.0
            qcfg = QueryConfig(maximum=trials, budget=query_budget,
                               threshold=safety, seed=int(seed),
                               compute_price=query_price)
            synthetic_context = make_scenario(qcfg.seed, scenario_id, qcfg.threshold)
            query_result = allocate(synthetic_context, qcfg, query_method)
            st.session_state.last_query = query_result
            compact = {key: query_result.get(key) for key in (
                "method", "action", "oracle", "regret", "truly_safe",
                "exploratory_action", "exploratory_truly_safe",
                "exploration_is_not_authorized", "used", "budget",
                "stop_reason", "lookahead_evaluations", "planning_seconds",
                "sampling_seconds", "last_information_per_query")}
            output = ("SIL / QUERY LAB — SYNTHETIC ONLY. Exploratory action "
                      "is NOT safety-certified.\n" + json.dumps(compact, indent=2)
                      + "\nUse /query-report for full auditable diagnostics.") 
        elif verb == "query-report":
            output = json.dumps(st.session_state.get("last_query", {}), indent=2)
        elif verb == "agents":
            output = json.dumps(lab.leaderboard(), indent=2)
        elif verb == "inspect":
            agent = lab.agents[int(parts[1]) - 1]
            output = json.dumps({"name": agent.name, "state": agent.state.tolist(),
                                 "posterior": agent.prior.tolist(), "alpha": agent.alpha,
                                 "beta": agent.beta}, indent=2)
        elif verb == "history":
            output = json.dumps(lab.history[-1] if lab.history else {}, indent=2)
        elif verb == "explain":
            output = explain({"round": lab.round, "board": lab.leaderboard(),
                              "most_recent": lab.history[-1] if lab.history else None})
        elif verb == "help":
            output = "/run [n] /query [scenario] [budget] [method] [compute_price] /query-report /agents /inspect N /history /explain /help — synthetic research only."
        else:
            output = "Unknown command. Use /help."
    except (ValueError, IndexError, TypeError) as exc:
        output = f"Invalid argument: {exc}"
    st.session_state.transcript.append({"role": "assistant", "content": output})
    st.rerun()
if lab.history:
    st.subheader("Current experimental results")
    latest = lab.history[-1]
    st.caption(f"Allocation: {latest['allocation_mode']} · Last round action-rollouts: "
               f"{latest['total_action_rollouts']:,} · "
               "Adaptive decision-gap stopping is heuristic, not a safety certificate.")
    st.dataframe(lab.leaderboard(), use_container_width=True, hide_index=True)
    if show_chart:
        import plotly.graph_objects as go
        fig = go.Figure()
        for agent in lab.agents:
            y = [row["cumulative_reward"] for record in lab.history
                 for row in record["agents"] if row["agent"] == agent.name]
            fig.add_scatter(x=list(range(1, len(y) + 1)), y=y, mode="lines", name=agent.name)
        fig.update_layout(title="Observed cumulative synthetic reward", xaxis_title="Round",
                          yaxis_title="Cumulative reward", height=300)
        st.plotly_chart(fig, use_container_width=True)
    st.download_button("Export deterministic audit JSON",
                       data=json.dumps({"seed": config.seed, "history": lab.history,
                                        "leaderboard": lab.leaderboard()}, indent=2),
                       file_name="sil-audit.json", mime="application/json")
