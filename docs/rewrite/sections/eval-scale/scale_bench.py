"""Evaluation at fleet sizes up to 32v32: compile, speed and memory per size, one process.

    PYTHONPATH=. uv run --no-sync python docs/rewrite/sections/eval-scale/scale_bench.py OUT.jsonl
        [--mode scripted|network] [--sizes 4x4 8x8 16x16 32x32 4x8] [--envs 4096]
        [--seconds 60] [--clock-ticks N] [--mixed]

Per size (in order, in one process, so a recompile at a new size shows as graphs added): the
first call's seconds (the compile when the size is the first; warm or cold by the inductor
cache), then a timed call of `--seconds` of play plus its drain: games played, games/s over
the whole call, game ticks/s (the played games' lengths), environment ticks/s while starting
(E x ticks before the cutoff over that time), peak GPU memory (allocated and reserved), and
Dynamo's graph and recompile counters after the size. `--mixed`: one more call with every
size in one queue. `network`: an untrained network (initial weights, seed 0) in seat A
against the yardstick; `scripted`: the yardstick against itself. One JSON line per call,
flushed as it is written.
"""

import argparse
import json
import subprocess
import time

import torch
from torch._dynamo.utils import counters

from lean import game, model, scripted
from lean.eval import Queue, play
from lean.eval.runner import SCRIPTED
from lean.scripted.parameters import vector


def queue_of(sizes: list[tuple[int, int]], games: int, network: bool) -> Queue:
    pick = torch.arange(games) % len(sizes)
    size = torch.tensor(sizes)[pick]
    zeros = torch.zeros(games, dtype=torch.long)
    net_a = zeros if network else zeros + SCRIPTED
    return Queue(zeros, zeros, torch.arange(games) % 2, size[:, 0], size[:, 1], net_a)


def graph_counters() -> dict:
    stats = counters["stats"]
    return {
        "unique_graphs": int(stats.get("unique_graphs", 0)),
        "frames_ok": int(stats.get("calls_captured", 0)),
        "recompiles": int(sum(counters["recompiles"].values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--mode", choices=("scripted", "network"), default="scripted")
    parser.add_argument("--sizes", nargs="+", default=["4x4", "8x8", "16x16", "32x32", "4x8"])
    parser.add_argument("--envs", type=int, nargs="+", default=[4096])
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--clock-ticks", type=int, default=None)
    parser.add_argument("--warm-games", type=int, default=64)
    parser.add_argument("--mixed", action="store_true")
    arguments = parser.parse_args()
    config = game.GameConfig()
    if arguments.clock_ticks is not None:
        config = game.GameConfig(clock_ticks=arguments.clock_ticks)
    device = "cuda"
    table = torch.tensor([vector(scripted.ScriptedConfig())], device=device)
    torch.manual_seed(0)
    policies = [model.Policy(model.ModelConfig.for_arena(config.arena_radius)).to(device).eval()]
    network = arguments.mode == "network"
    profile = subprocess.run(["powerprofilesctl", "get"], capture_output=True, text=True).stdout
    out = open(arguments.out, "a")

    def record(entry):
        entry.update(
            mode=arguments.mode,
            profile=profile.strip(),
            clock=time.strftime("%H:%M:%S"),
            **graph_counters(),
        )
        out.write(json.dumps(entry) + "\n")
        out.flush()
        print(json.dumps(entry), flush=True)

    sizes = [tuple(int(part) for part in text.split("x")) for text in arguments.sizes]
    runs = [[size] for size in sizes] + ([sizes] if arguments.mixed else [])
    for envs in arguments.envs:
        for run in runs:
            label = "+".join(f"{a}x{b}" for a, b in run)
            # The first call: compiles when nothing has (a short queue, a short drain).
            started = time.perf_counter()
            play(
                config, queue_of(run, arguments.warm_games, network), table,
                policies if network else (), envs=arguments.warm_games,
                minimum_envs=arguments.warm_games, compiled=True,
            )  # fmt: skip
            torch.cuda.synchronize()
            first_call = time.perf_counter() - started
            torch.cuda.reset_peak_memory_stats()
            ticks = []

            def progress(games, busy, seconds, ticks=ticks):
                ticks.append((seconds, games))

            games = 64 * envs  # more than the cutoff lets start
            started = time.perf_counter()
            try:
                outcome = play(
                    config, queue_of(run, games, network), table, policies if network else (),
                    envs=envs, compiled=True, seconds=arguments.seconds, progress=progress,
                    seed=1,
                )  # fmt: skip
            except torch.OutOfMemoryError as error:
                record({"sizes": label, "envs": envs, "oom": str(error)[:200]})
                torch.cuda.empty_cache()
                continue
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            played = outcome.result >= 0
            starting = [
                (seconds, count) for seconds, count in ticks if seconds <= arguments.seconds
            ]
            segments_before = len(starting)
            record(
                {
                    "sizes": label,
                    "envs": envs,
                    "first_call_s": round(first_call, 1),
                    "games": int(played.sum()),
                    "seconds": round(elapsed, 1),
                    "games_per_s": round(int(played.sum()) / elapsed, 1),
                    "game_ticks_per_s": round(int(outcome.length.sum()) / elapsed),
                    "env_ticks_per_s_starting": round(segments_before * 32 * envs / starting[-1][0])
                    if starting
                    else None,
                    "mean_length": round(float(outcome.length[played].float().mean()), 1),
                    "a_score": round(
                        float(
                            (
                                (outcome.result[played] == 0).float()
                                + 0.5 * (outcome.result[played] == 2).float()
                            ).mean()
                        ),
                        4,
                    ),  # fmt: skip
                    "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                    "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
                }
            )


if __name__ == "__main__":
    main()
