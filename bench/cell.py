"""One benchmark cell in its own process; prints `RESULT {json}`.

train: build the trainer (compile), run one episode of rollouts to reach a steady state
(staggered clocks, combat under way), then time rollouts and updates. A train cell may name
a training preset instead of a size ("preset": a game preset, or "mixed").
realtime: CPU, one match, the latency of one tick.
"""

import json
import math
import os
import sys
import time

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

import torch  # noqa: E402

from bnb.contracts import GameConfig  # noqa: E402
from bnb.contracts.action import NUM_POWER, NUM_SHOOT, NUM_STICK  # noqa: E402

TIMED = 3


def train_config(size: dict | None, environments: int, preset: str | None):
    """The training config of a cell: a size, a game preset or the mixed preset."""
    from bnb.game.modes.rules import PRESETS
    from bnb.runtime.train import TrainConfig, mixed

    if preset == "mixed":
        game, pools = mixed()
        return TrainConfig(game=game, pools=pools, environments=environments)
    game = PRESETS[preset] if preset is not None else GameConfig(**size)
    return TrainConfig(game=game, environments=environments)


def train_cell(size: dict | None, environments: int, preset: str | None = None) -> dict:
    """Compile, run one episode, then the best of `TIMED` collects and updates."""
    from bnb.runtime import metrics
    from bnb.runtime.train import Trainer

    torch.manual_seed(0)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    config = train_config(size, environments, preset)
    started = time.perf_counter()
    trainer = Trainer(config, device)
    trainer.update()  # compiles the rollout stages, the graph and the learner
    torch.cuda.synchronize()
    compile_seconds = time.perf_counter() - started
    episode_ticks = max(pool.episode_ticks for pool in trainer.rollout.rules.pools)
    for _ in range(math.ceil(episode_ticks / config.length)):  # one episode
        trainer.collect()
    rollout_seconds, learn_seconds = [], []
    for _ in range(TIMED):
        torch.cuda.synchronize()
        started = time.perf_counter()
        store, bootstraps = trainer.collect()
        torch.cuda.synchronize()
        collected = time.perf_counter()
        metrics.read(trainer.ppo.update(store, bootstraps))  # the update's one readback
        learned = time.perf_counter()
        rollout_seconds.append(collected - started)
        learn_seconds.append(learned - collected)
    ticks = environments * config.length
    rollout, learn = min(rollout_seconds), min(learn_seconds)
    # The keys are bench/results.jsonl's format, kept so its history compares.
    return dict(
        compile_s=compile_seconds,
        rollout_ms=rollout * 1e3,
        learn_ms=learn * 1e3,
        rollout_env_ticks_per_s=ticks / rollout,
        train_env_ticks_per_s=ticks / (rollout + learn),
        peak_vram_gb=torch.cuda.max_memory_allocated(device) / 2**30,
        counters=trainer.rollout.counters(),
    )


def realtime_cell(size: dict, steps: int = 1000) -> dict:
    """The CPU tick latency of one match at this size: p50 and p99 over `steps` ticks."""
    from bnb.runtime.realtime import Realtime

    torch.manual_seed(0)
    torch.set_num_threads(min(6, torch.get_num_threads()))  # an entry point's setting
    realtime = Realtime(GameConfig(**size))
    ships = 2 * size["ships_per_team"]
    started = time.perf_counter()
    realtime.step(torch.zeros(ships, 3, dtype=torch.int32))
    compile_seconds = time.perf_counter() - started
    sizes = torch.tensor([NUM_POWER, NUM_STICK, NUM_SHOOT], dtype=torch.float32)
    times = []
    for index in range(300 + steps):
        actions = (torch.rand(ships, 3) * sizes).to(torch.int32)
        started = time.perf_counter()
        realtime.step(actions)
        if index >= 300:
            times.append(time.perf_counter() - started)
    times.sort()
    return dict(
        compile_s=compile_seconds,
        p50_ms=times[len(times) // 2] * 1e3,
        p99_ms=times[int(0.99 * len(times))] * 1e3,
    )


if __name__ == "__main__":
    cell = json.loads(sys.argv[1])
    if cell["kind"] == "train":
        result = train_cell(cell.get("size"), cell["envs"], cell.get("preset"))
    else:
        result = realtime_cell(cell["size"])
    print("RESULT " + json.dumps(dict(cell=cell, **result)), flush=True)
