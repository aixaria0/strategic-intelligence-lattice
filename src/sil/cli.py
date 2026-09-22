"""Terminal-first, no-GUI command interface for synthetic research."""
import argparse
import json
from pathlib import Path
from .core import Config
from .engine import Tournament

HELP = """/help | /run [n] | /agents | /inspect N | /history | /reset | /quit
All values are synthetic. Actions are simulated, never executed externally."""

def render(board):
    print(f"{'AGENT':<12} {'TOTAL REWARD':>13} {'NET UPLIFT':>12} {'RISK EVENTS':>13}")
    for row in board:
        print(f"{row['name']:<12} {row['cumulative_reward']:>13.4f} "
              f"{row['cumulative_uplift']:>12.4f} {row['safety_violations']:>13}")

def chat(config: Config, num_agents: int, evolve: bool,
         allocation_mode: str = "fixed", simulation_budget: int | None = None):
    lab = Tournament(config, num_agents, evolve, allocation_mode, simulation_budget)
    print("SIL / terminal research console. " + HELP)
    while True:
        try:
            command = input("sil> ").strip().split()
        except (EOFError, KeyboardInterrupt):
            print("\nSession closed.")
            return
        if not command:
            continue
        verb = command[0].lower().lstrip("/")
        try:
            if verb in ("quit", "exit"):
                return
            if verb == "help":
                print(HELP)
            elif verb == "run":
                count = int(command[1]) if len(command) > 1 else 1
                lab.run(count)
                print(f"Completed synthetic round {lab.round}.")
                render(lab.leaderboard())
            elif verb == "agents":
                render(lab.leaderboard())
            elif verb == "inspect":
                agent = lab.agents[int(command[1]) - 1]
                print(json.dumps({"name": agent.name, "state": agent.state.tolist(),
                                  "model_weights": agent.prior.tolist(), "alpha": agent.alpha,
                                  "beta": agent.beta}, indent=2))
            elif verb == "history":
                print(json.dumps(lab.history[-1] if lab.history else {}, indent=2))
            elif verb == "reset":
                lab = Tournament(config, num_agents, evolve, allocation_mode, simulation_budget)
                print("Synthetic simulation reset.")
            else:
                print("Unknown command. /help")
        except (ValueError, IndexError) as exc:
            print(f"Invalid argument: {exc}")

def main():
    parser = argparse.ArgumentParser(description="SIL: synthetic strategy simulation")
    parser.add_argument("mode", nargs="?", choices=["run", "chat", "query-lab"], default="chat")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--trials", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-evolution", action="store_true")
    parser.add_argument("--allocation", choices=["fixed", "adaptive"], default="fixed")
    parser.add_argument("--budget", type=int, help="action-rollout budget for adaptive mode; default=fixed cap")
    parser.add_argument("--scenario", type=int, default=0,
                        help="query-lab context id")
    parser.add_argument("--method", choices=["fixed", "uniform", "random", "adaptive", "hybrid", "evsi_reward", "c_voc", "c_voc_governor", "certificate_portfolio", "certificate_targeted"],
                        default="uniform", help="query-lab policy (uniform evidence-based baseline; ACA/EVSI experimental)")
    parser.add_argument("--risk-threshold", type=float, default=0.75,
                        help="query-lab minimum safety probability under each model")
    parser.add_argument("--delta", type=float, default=0.05,
                        help="query-lab familywise failure budget")
    parser.add_argument("--compute-price", type=float, default=0.0,
                        help="explicit reward-unit price per query for opt-in c_voc_governor")
    parser.add_argument("--output", type=Path, help="JSON audit snapshot path")
    args = parser.parse_args()
    if args.mode == "query-lab":
        from .query_lab import QueryConfig, allocate, make_scenario
        query_cfg = QueryConfig(
            budget=args.budget if args.budget is not None else 9 * args.trials,
            maximum=args.trials, seed=args.seed, threshold=args.risk_threshold,
            delta=args.delta, compute_price=args.compute_price,
        )
        scenario = make_scenario(args.seed, args.scenario, query_cfg.threshold)
        result = allocate(scenario, query_cfg, args.method)
        print(json.dumps(result, indent=2))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return
    cfg = Config(trials=args.trials, horizon=args.horizon, seed=args.seed)
    if args.mode == "chat":
        chat(cfg, args.agents, not args.no_evolution, args.allocation, args.budget)
        return
    lab = Tournament(cfg, args.agents, not args.no_evolution, args.allocation, args.budget)
    lab.run(args.rounds)
    render(lab.leaderboard())
    print(f"Allocation: {args.allocation}; total action-rollouts: "
          f"{sum(r['total_action_rollouts'] for r in lab.history)}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"config": vars(cfg), "history": lab.history,
                                          "leaderboard": lab.leaderboard()}, indent=2) + "\n",
                               encoding="utf-8")
        print(f"Saved {args.output}")

if __name__ == "__main__":
    main()
