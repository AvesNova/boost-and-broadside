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

TIMED = 3


def train_config(size: dict | None, envs: int, preset: str | None):
    from bnb.game.modes.rules import PRESETS
    from bnb.runtime.train import TrainConfig, mixed

    if preset == "mixed":
        game, pools = mixed()
        return TrainConfig(game=game, pools=pools, num_envs=envs)
    game = PRESETS[preset] if preset is not None else GameConfig(**size)
    return TrainConfig(game=game, num_envs=envs)


def train_cell(size: dict | None, envs: int, preset: str | None = None) -> dict:
    from bnb.runtime.train import Trainer

    torch.manual_seed(0)
    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(dev)
    cfg = train_config(size, envs, preset)
    t0 = time.perf_counter()
    trainer = Trainer(cfg, dev)
    trainer.update()  # compiles the rollout stages, the graph and the learner
    torch.cuda.synchronize()
    compile_s = time.perf_counter() - t0
    episode_ticks = max(pool.episode_ticks for pool in trainer.rollout.rules.pools)
    for _ in range(math.ceil(episode_ticks / cfg.length)):  # one episode
        trainer.collect()
    rollout_s, learn_s = [], []
    for _ in range(TIMED):
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        traj, last_value = trainer.collect()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        trainer.ppo.update(traj, last_value)
        t3 = time.perf_counter()
        rollout_s.append(t2 - t1)
        learn_s.append(t3 - t2)
    ticks = envs * cfg.length
    r, lrn = min(rollout_s), min(learn_s)
    return dict(compile_s=compile_s, rollout_ms=r * 1e3, learn_ms=lrn * 1e3,
                rollout_env_ticks_per_s=ticks / r, train_env_ticks_per_s=ticks / (r + lrn),
                peak_vram_gb=torch.cuda.max_memory_allocated(dev) / 2**30,
                counters=trainer.rollout.counters())


def realtime_cell(size: dict, steps: int = 1000) -> dict:
    from bnb.runtime.realtime import Realtime

    torch.manual_seed(0)
    torch.set_num_threads(min(6, torch.get_num_threads()))  # an entry point's setting
    rt = Realtime(GameConfig(**size))
    N = 2 * size["ships_per_team"]
    t0 = time.perf_counter()
    rt.step(torch.zeros(N, 3, dtype=torch.int32))
    compile_s = time.perf_counter() - t0
    times = []
    for i in range(300 + steps):
        a = (torch.rand(N, 3) * torch.tensor([3.0, 7.0, 2.0])).to(torch.int32)
        t1 = time.perf_counter()
        rt.step(a)
        if i >= 300:
            times.append(time.perf_counter() - t1)
    times.sort()
    return dict(compile_s=compile_s, p50_ms=times[len(times) // 2] * 1e3,
                p99_ms=times[int(0.99 * len(times))] * 1e3)


if __name__ == "__main__":
    cell = json.loads(sys.argv[1])
    fn = train_cell if cell["kind"] == "train" else realtime_cell
    args = (cell.get("size"), cell["envs"], cell.get("preset")) if cell["kind"] == "train" \
        else (cell["size"],)
    print("RESULT " + json.dumps(dict(cell=cell, **fn(*args))), flush=True)
