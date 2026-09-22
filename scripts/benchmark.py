"""Repeatable wall-clock microbenchmark; report machine-specific results."""
from time import perf_counter
import argparse
from sil.core import Config
from sil.engine import Tournament

p = argparse.ArgumentParser()
p.add_argument('--agents', type=int, default=8)
p.add_argument('--trials', type=int, default=128)
p.add_argument('--horizon', type=int, default=8)
p.add_argument('--rounds', type=int, default=3)
a = p.parse_args()
lab = Tournament(Config(trials=a.trials, horizon=a.horizon), a.agents, evolve=False)
t0 = perf_counter()
lab.run(a.rounds)
seconds = perf_counter() - t0
print(f'agents={a.agents} trials={a.trials} horizon={a.horizon} rounds={a.rounds} wall_seconds={seconds:.4f}')
print(f'agent_rounds_per_second={a.agents*a.rounds/seconds:.2f}')
