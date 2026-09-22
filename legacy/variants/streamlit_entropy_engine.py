# HISTORICAL FIRST STREAMLIT PROTOTYPE — includes known modeling errors.
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Strategic Entropy Engine", layout="wide")
st.title("🧠 Strategic Entropy Engine Prototype")
st.sidebar.header("Human Administrator")
num_actions = st.sidebar.slider("Candidate Actions", 2, 10, 5)
num_adversaries = st.sidebar.slider("Adversary Hypotheses", 1, 5, 2)
entropy_alpha = st.sidebar.slider("Alpha (favor basin reduction)", 0.0, 2.0, 1.0)
entropy_beta = st.sidebar.slider("Beta (adversary variance)", 0.0, 2.0, 1.0)
steps = st.sidebar.slider("Monte Carlo Steps per rollout", 5, 50, 10)
st.sidebar.markdown("---")
st.sidebar.markdown("Adjust mission constraints in code or future UI.")
weights = np.ones(num_adversaries) / num_adversaries

def monte_carlo_rollout(state, action, adversary_idx, steps=10):
    trajectory = [state]
    s = state
    for t in range(steps):
        adversary_effect = np.random.normal(loc=0.0, scale=(adversary_idx+1)*0.1)
        s = s + action - adversary_effect
        trajectory.append(s)
    return np.array(trajectory)

def compute_entropy(traj):
    return np.var(traj)

if st.button("Run Simulation"):
    state_0 = np.random.uniform(0,1)
    action_scores = []
    for action_idx in range(num_actions):
        weighted_entropy = 0
        for adv_idx in range(num_adversaries):
            traj = monte_carlo_rollout(state_0, action_idx, adv_idx, steps)
            entropy = compute_entropy(traj)
            weighted_entropy += weights[adv_idx] * entropy
        action_scores.append(weighted_entropy)
    best_action_idx = np.argmax([entropy_beta*score - entropy_alpha*score for score in action_scores])
    st.write(f"Initial State: {state_0:.3f}")
    st.write(f"Action Scores: {np.round(action_scores, 3)}")
    st.write(f"Selected Action: {best_action_idx}")
    fig, ax = plt.subplots(figsize=(8,4))
    for adv_idx in range(num_adversaries):
        traj = monte_carlo_rollout(state_0, best_action_idx, adv_idx, steps)
        ax.plot(range(steps+1), traj, label=f"Adversary {adv_idx+1}")
    ax.set_xlabel("Step")
    ax.set_ylabel("State")
    ax.set_title("Monte Carlo Trajectories (Entropy Basins)")
    ax.legend()
    st.pyplot(fig)
    observation = np.random.normal(loc=best_action_idx, scale=0.2)
    likelihoods = np.exp(-0.5 * ((observation - np.arange(num_adversaries))**2) / 0.1)
    weights = likelihoods * weights
    weights /= np.sum(weights)
    st.write(f"Updated Adversary Weights: {np.round(weights,3)}")
