# HISTORICAL tournament prototype; contains fake entropy, repeated evaluations and invalid regret.
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Entropy Tournament", layout="wide")
st.title("🧠 Entropy Steering AI Tournament")
st.sidebar.header("Tournament Setup")
num_agents = st.sidebar.slider("Number of Agents", 2, 6, 3)
num_adversaries = st.sidebar.slider("Adversary Hypotheses", 1, 5, 2)
mc_steps = st.sidebar.slider("Monte Carlo Steps", 5, 30, 10)
rounds = st.sidebar.slider("Rounds", 1, 5, 3)

class Agent:
    def __init__(self, name, alpha=1.0, beta=1.0):
        self.name = name
        self.alpha = alpha
        self.beta = beta
        self.weights = np.ones(num_adversaries) / num_adversaries
        self.regret = 0
    def monte_carlo_rollout(self, state, action, adv_idx):
        traj = [state]
        s = state
        for t in range(mc_steps):
            adversary_effect = np.random.normal(loc=0.0, scale=(adv_idx+1)*0.1)
            s = s + action - adversary_effect
            traj.append(s)
        return np.array(traj)
    def select_action(self, state):
        action_scores = []
        for action_idx in range(num_agents):
            weighted_entropy = 0
            for adv_idx in range(num_adversaries):
                traj = self.monte_carlo_rollout(state, action_idx, adv_idx)
                entropy = np.var(traj)
                weighted_entropy += self.weights[adv_idx] * entropy
            action_scores.append(weighted_entropy)
        final_scores = [self.beta*s - self.alpha*s for s in action_scores]
        return int(np.argmax(final_scores)), action_scores
    def update_adversary_weights(self, observation):
        likelihoods = np.exp(-0.5 * ((observation - np.arange(num_adversaries))**2) / 0.1)
        self.weights *= likelihoods
        self.weights /= np.sum(self.weights)

agents = [Agent(f"Agent {i+1}", alpha=np.random.rand()+0.5, beta=np.random.rand()+0.5) for i in range(num_agents)]
state_0 = np.random.uniform(0,1)
if st.button("Run Tournament"):
    st.write(f"Initial State: {state_0:.3f}")
    leaderboard = []
    for rnd in range(1, rounds+1):
        st.write(f"### Round {rnd}")
        round_results = {}
        for agent in agents:
            action, scores = agent.select_action(state_0)
            observation = np.random.normal(loc=action, scale=0.2)
            agent.update_adversary_weights(observation)
            round_results[agent.name] = (action, np.round(scores,3))
            agent.regret += np.max(scores) - scores[action]
        leaderboard.append({a.name: a.regret for a in agents})
        st.write(round_results)
        fig, ax = plt.subplots(figsize=(10,5))
        for agent in agents:
            best_action, _ = agent.select_action(state_0)
            for adv_idx in range(num_adversaries):
                traj = agent.monte_carlo_rollout(state_0, best_action, adv_idx)
                ax.plot(range(mc_steps+1), traj, label=f"{agent.name} Adv {adv_idx+1}")
        ax.set_xlabel("Step")
        ax.set_ylabel("State")
        ax.set_title(f"Monte Carlo Trajectories – Round {rnd}")
        ax.legend(fontsize=8)
        st.pyplot(fig)
    st.write("## Tournament Leaderboard (Cumulative Regret)")
    leaderboard_arr = np.array([[l[a.name] for a in agents] for l in leaderboard])
    for idx, agent in enumerate(agents):
        st.write(f"{agent.name}: {np.sum(leaderboard_arr[:,idx]):.3f}")
