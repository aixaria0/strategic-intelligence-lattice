# HISTORICAL HTTP worker sketch; a single random trajectory is not a full MC cluster.
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

@app.route("/rollout", methods=["POST"])
def rollout():
    data = request.json
    state = data.get("state", 0.5)
    action = data.get("action", 1)
    adv_idx = data.get("adv_idx", 0)
    steps = data.get("steps", 10)
    traj = [state]
    s = state
    for t in range(steps):
        adversary_effect = np.random.normal(0, 0.1*(adv_idx+1))
        s = s + action - adversary_effect
        traj.append(s)
    return jsonify({"trajectory": traj, "entropy": np.var(traj)})

@app.route("/bayes_update", methods=["POST"])
def bayes_update():
    data = request.json
    weights = np.array(data.get("weights"))
    observation = data.get("observation")
    num_adversaries = len(weights)
    likelihoods = np.exp(-0.5*((observation - np.arange(num_adversaries))**2)/0.1)
    weights *= likelihoods
    weights /= np.sum(weights)
    return jsonify({"updated_weights": weights.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
