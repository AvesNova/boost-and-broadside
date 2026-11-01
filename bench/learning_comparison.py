"""A short learning comparison: Yemong (or the toy model) trained against the chaser, one
JSON line per update; prints `RESULT {json}` at the end.

    uv run --no-sync python -m bench.learning_comparison OUT.jsonl '{"preset": "deathmatch",
        "envs": 128, "updates": 300, "learner": {...}, "precision": "float32", "seed": 0}'

Each line: the update's losses, KL, clip fraction, learn and rollout seconds, and the
learner's results in that update's segments (wins, draws, losses of finished episodes).
Every `evaluate_every` updates (and at the end) the win rate over `games` fresh games
against the chaser. Options: "model" ("yemong" or "relational"), "precision", "learner"
(LearnerConfig fields), "seed". Takes the benchmark lock.
"""

import json
import os
import sys
import time

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

import torch  # noqa: E402

from bench.lock import bench_lock  # noqa: E402


def main() -> None:
    from bnb.contracts import Pool
    from bnb.game.modes.rules import PRESETS
    from bnb.learn.ppo import LearnerConfig
    from bnb.runtime.train import LEARNER, TrainConfig, Trainer, set_precision, win_rate

    path, cell = sys.argv[1], json.loads(sys.argv[2])
    seed, precision = cell.get("seed", 0), cell.get("precision", "float32")
    every, games = cell.get("evaluate_every", 50), cell.get("games", 256)
    with bench_lock():
        torch.manual_seed(seed)
        set_precision(precision)
        cfg = TrainConfig(game=PRESETS[cell["preset"]], num_envs=cell["envs"],
                          model=cell.get("model", "yemong"), precision=precision,
                          pools=(Pool("chaser", 1.0, (LEARNER, "chaser")),),
                          learner=LearnerConfig(**dict(cell.get("learner", {}), seed=seed)))
        trainer = Trainer(cfg, torch.device("cuda"))
        evaluations = []
        with open(path, "w") as out:
            for update in range(1, cell["updates"] + 1):
                stats = trainer.update()
                results = stats.pop("pools")["chaser"]["results"]
                line = {key: value for key, value in stats.items()
                        if isinstance(value, (int, float))}
                line.update(results)
                if update % every == 0 or update == cell["updates"]:
                    started = time.perf_counter()
                    line["evaluation"] = win_rate(trainer, games=games)
                    line["evaluation_s"] = time.perf_counter() - started
                    evaluations.append((update, line["evaluation"]))
                out.write(json.dumps(line) + "\n")
                out.flush()
    print("RESULT " + json.dumps(dict(cell=cell, evaluations=evaluations)), flush=True)


if __name__ == "__main__":
    main()
