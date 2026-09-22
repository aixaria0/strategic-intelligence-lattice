# HISTORICAL: named a Core LLM service but did NOT call an LLM.
from flask import Flask, request, jsonify
app = Flask(__name__)
@app.route("/reframe", methods=["POST"])
def reframe():
    summary = request.json
    insight = f"Strategic insight: Favor trajectories with max pressure {max(summary.get('pressures', [0]))}"
    return jsonify({"insight": insight})
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
