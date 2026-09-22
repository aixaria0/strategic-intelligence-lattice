# HISTORICAL SKETCH: this uses path variance as a false general entropy proxy.
import numpy as np

S = np.array([1.0, 1.0, 0.0])

def transition(S, action, adv_action, noise_scale=0.1):
    noise = np.random.normal(0, noise_scale, size=S.shape)
    return S + action - adv_action + noise

def value(S):
    return S[0] - S[1] + 0.5 * S[2]

def is_favorable(S):
    return value(S) > 0

def rollout_entropy(S, action, adv_action, steps=20, trials=50):
    trajectories = []
    for _ in range(trials):
        s = S.copy()
        for _ in range(steps):
            s = transition(s, action, adv_action)
        trajectories.append(s)
    trajectories = np.array(trajectories)
    return np.var(trajectories, axis=0).mean()

def steering_score(S, action, adv_action):
    entropy = rollout_entropy(S, action, adv_action)
    s_next = transition(S, action, adv_action, noise_scale=0.0)
    if is_favorable(s_next):
        return -entropy
    return entropy

# No claims of inevitability, safety or profitable real-world decisions follow.
