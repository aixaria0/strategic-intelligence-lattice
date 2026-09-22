# HISTORICAL SKETCH recovered from the conversation; NOT the canonical engine.
# The score cancels when alpha == beta, and this mock 'regret' uses the wrong comparator.
import numpy as np

class Agent:
    def __init__(self, name, alpha=1.0, beta=1.0, num_adversaries=2):
        self.name, self.alpha, self.beta = name, alpha, beta
        self.weights = np.ones(num_adversaries) / num_adversaries
        self.regret = 0

    def monte_carlo_rollout(self, state, action, adv_idx, mc_steps=10):
        traj, s = [state], state
        for t in range(mc_steps):
            adversary_effect = np.random.normal(0, (adv_idx+1)*0.1)
            s = s + action - adversary_effect
            traj.append(s)
        return np.array(traj)

    def select_action(self, state, num_actions=3):
        scores = []
        for action_idx in range(num_actions):
            weighted_entropy = 0
            for adv_idx in range(len(self.weights)):
                traj = self.monte_carlo_rollout(state, action_idx, adv_idx)
                weighted_entropy += self.weights[adv_idx] * np.var(traj)
            scores.append(weighted_entropy)
        final_scores = [self.beta*s - self.alpha*s for s in scores]
        return int(np.argmax(final_scores)), scores

    def update_adversary_weights(self, observation):
        likelihoods = np.exp(-0.5 * ((observation - np.arange(len(self.weights)))**2) / 0.1)
        self.weights *= likelihoods
        self.weights /= np.sum(self.weights)

# Earlier iterations reinitialized agent instances on every Streamlit rerun.
# Earlier 'regret' used max(raw_variance) - chosen_raw_variance, not objective regret.
