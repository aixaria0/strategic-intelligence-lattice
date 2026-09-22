# HISTORICAL sequential HTTP coordinator, not a concurrent distributed planner.
import streamlit as st
import requests
import numpy as np

st.title("Distributed Entropy Tournament Coordinator")
worker_urls = st.sidebar.text_area("Worker URLs (comma-separated)", "http://worker1:5000,http://worker2:5000")
worker_urls = [url.strip() for url in worker_urls.split(",")]
num_actions = st.sidebar.slider("Candidate Actions", 2, 10, 5)
steps = st.sidebar.slider("Monte Carlo Steps", 5, 30, 10)
if st.button("Run Distributed Round"):
    state_0 = np.random.uniform(0,1)
    st.write(f"Initial State: {state_0:.3f}")
    results = []
    for action in range(num_actions):
        action_entropy = 0
        for adv_idx in range(len(worker_urls)):
            worker_url = worker_urls[adv_idx % len(worker_urls)]
            r = requests.post(worker_url + "/rollout",
                              json={"state": state_0, "action": action,
                                    "adv_idx": adv_idx, "steps": steps})
            res = r.json()
            action_entropy += res["entropy"]
        results.append(action_entropy)
    best_action = int(np.argmax(results))
    st.write(f"Action Entropy Scores: {np.round(results,3)}")
    st.write(f"Selected Action: {best_action}")
