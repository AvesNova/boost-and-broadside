"""An Elo ladder in one job: wall time and ratings, for the routed runner against the tree
before it (pairing-by-pairing calls).

    PYTHONPATH=TREE uv run --no-sync python docs/rewrite/sections/eval-scale/ladder_bench.py
        OUT_DIR LABEL [--games 2048 [128 ...]] [--envs 4096] [--networks 3]

Agents: `--networks` untrained networks (initial weights, seeds 0, 1, ...; saved as
checkpoints in OUT_DIR) and four scripted agents (the yardstick and three variants). Every
pair plays `--games` games at 4v4 (round robin) through Evaluator.ensure in one job, so the
tree on PYTHONPATH decides how the job is split into runner calls. A warm-up job first
(every layout the job uses, a throwaway store) so that the timed job is compile-free. Writes
OUT_DIR/LABEL-GAMES.json per game count (a fresh store each): seconds, games, runner calls,
Elo per agent (name, rating, error).
"""

import argparse
import itertools
import json
import subprocess
import time
from pathlib import Path

import torch

from lean import game, model, scripted
from lean.eval import Evaluator, NetworkAgent, ScriptedAgent, Store
from lean.eval import runner as runner_module
from lean.learn import LearnConfig, WeightAverages, save_checkpoint

VARIANTS = {
    "yardstick": {},
    "fight_speed_1.3": {"fight_speed": 1.3},
    "fight_speed_0.8": {"fight_speed": 0.8},
    "turn_0.003": {"turn": 0.003},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("label")
    parser.add_argument("--games", type=int, nargs="+", default=[2048])
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--networks", type=int, default=3)
    arguments = parser.parse_args()
    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    profile = subprocess.run(["powerprofilesctl", "get"], capture_output=True, text=True).stdout
    names = {}
    agents = []
    for seed in range(arguments.networks):
        path = out / f"untrained{seed}.pt"
        if not path.exists():
            torch.manual_seed(seed)
            policy = model.Policy(model.ModelConfig())
            save_checkpoint(path, policy, WeightAverages(policy, LearnConfig()), LearnConfig())
        agent = NetworkAgent(str(path), "live")
        agents.append(agent)
        names[agent.id] = f"untrained{seed}"
    for name, values in VARIANTS.items():
        agent = ScriptedAgent.of(scripted.ScriptedConfig().replace(**values))
        agents.append(agent)
        names[agent.id] = name

    calls = []
    real = runner_module.play

    def counting(*positional, **keywords):
        calls.append(len(positional[1]))
        return real(*positional, **keywords)

    runner_module.play = counting
    config = game.GameConfig()
    sizes = (4, 4)
    pairs = list(itertools.combinations(agents, 2))

    warm_store = out / f"{arguments.label}-warm.sqlite"
    warm_store.unlink(missing_ok=True)
    started = time.perf_counter()
    Evaluator(Store(warm_store), config, envs=256).ensure([(a, b, sizes, 256) for a, b in pairs])
    torch.cuda.synchronize()
    warm = time.perf_counter() - started
    calls.clear()

    for per_pairing in arguments.games:
        calls.clear()
        path = out / f"{arguments.label}-{per_pairing}.sqlite"
        path.unlink(missing_ok=True)
        store = Store(path)
        evaluator = Evaluator(store, config, envs=arguments.envs)

        def progress(games, busy, seconds):
            print(
                f"{time.strftime('%H:%M:%S')} {games} started {busy} busy {seconds:.0f} s",
                flush=True,
            )

        evaluator.progress = progress
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        evaluator.ensure([(a, b, sizes, per_pairing) for a, b in pairs])
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        ratings = [
            (names[identity], round(rating, 1), round(error, 1))
            for identity, rating, error in evaluator.elo(sizes)
        ]
        report = {
            "label": arguments.label,
            "per_pairing": per_pairing,
            "profile": profile.strip(),
            "warm_up_s": round(warm, 1),
            "seconds": round(seconds, 1),
            "games": store.total_games(),
            "games_per_s": round(store.total_games() / seconds, 1),
            "runner_calls": calls,
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
            "elo": ratings,
            "finished": time.strftime("%H:%M:%S"),
        }
        (out / f"{arguments.label}-{per_pairing}.json").write_text(json.dumps(report, indent=1))
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
