"""One process of the exact-resume test (runs acceptance 3), as an entry point would run
it: seed the process, then train straight through, or train and save, or resume and train.

    python resume_worker.py MODE DIRECTORY DEVICE MODEL UPDATES

MODE: "straight" (2 x UPDATES updates), "first" (UPDATES updates, then a training
checkpoint at DIRECTORY/checkpoint) or "second" (resume from it, UPDATES more). Each writes
its metrics and its final weights into DIRECTORY/<MODE>.
"""

import json
import os
import pathlib
import sys

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")  # as the entry points set it

import safetensors.torch  # noqa: E402
import torch  # noqa: E402

from bnb.contracts import GameConfig  # noqa: E402
from bnb.learn.ppo import LearnerConfig  # noqa: E402
from bnb.runs import checkpoint  # noqa: E402
from bnb.runtime.train import TrainConfig, Trainer, set_precision  # noqa: E402

TIMES = ("rollout_s", "learn_s")


def config(model: str, device: str) -> TrainConfig:
    learner = LearnerConfig(epochs=2, minibatches=2, warmup_steps=1000, micro_batch_rows=64)
    return TrainConfig(game=GameConfig(2, 2, 1500.0, episode_s=4.0, respawn_s=1.0),
                       num_envs=16, updates=0, length=32, model=model, relational_hidden=32,
                       league_every=1, learner=learner, compile=device == "cuda")


def main() -> None:
    mode, directory, device, model, updates = sys.argv[1:]
    directory, device, updates = pathlib.Path(directory), torch.device(device), int(updates)
    cfg = config(model, device.type)
    torch.manual_seed(cfg.seed)
    set_precision(cfg.precision)
    if mode == "second":
        trainer = checkpoint.load_training(directory / "checkpoint", device)
        checkpoint.restore_default_generators(directory / "checkpoint", device)
    else:
        trainer = Trainer(cfg, device)
    count = 2 * updates if mode == "straight" else updates
    rows = []
    for _ in range(count):
        row = trainer.update()
        rows.append({name: value for name, value in row.items() if name not in TIMES})
    if mode == "first":
        checkpoint.save_training(directory / "checkpoint", trainer)
    out = directory / mode
    out.mkdir(parents=True)
    (out / "metrics.json").write_text(json.dumps(rows))
    safetensors.torch.save_file(checkpoint.weights(trainer.model), out / "weights.safetensors")


if __name__ == "__main__":
    main()
