"""One process of the exact-resume test (runs acceptance 3), as an entry point would run
it: seed the process, then train straight through, or train and save, or resume and train.

    python resume_worker.py MODE DIRECTORY DEVICE MODEL UPDATES [DAMAGE]

MODE: "straight" (2 x UPDATES updates), "first" (UPDATES updates, then a training
checkpoint at DIRECTORY/checkpoint) or "second" (resume from it, UPDATES more). Each writes
its metrics and its final weights into DIRECTORY/<MODE>. The league slot is copied every
LEAGUE_EVERY updates, so at the checkpoint it lags the model. DAMAGE (with "second", to
show the test sees it): "league" adds noise to the league slot's weights after the restore,
"return_scale" puts back a fresh trainer's return scale.
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
LEAGUE_EVERY = 3


def config(model: str, device: str) -> TrainConfig:
    # A learning rate that moves the model visibly in two updates, and a return-scale floor
    # under the returns' spread, so both the league's lag and the return scale show.
    learner = LearnerConfig(epochs=2, minibatches=2, learning_rate=3e-3, warmup_steps=1000,
                            micro_batch_rows=64, return_scale_floor=1e-7)
    return TrainConfig(game=GameConfig(2, 2, 1500.0, episode_s=4.0, respawn_s=1.0),
                       num_envs=16, updates=0, length=32, model=model, relational_hidden=32,
                       league_every=LEAGUE_EVERY, learner=learner, compile=device == "cuda")


def main() -> None:
    mode, directory, device, model, updates, *damage = sys.argv[1:]
    directory, device, updates = pathlib.Path(directory), torch.device(device), int(updates)
    settings = config(model, device.type)
    torch.manual_seed(settings.seed)
    set_precision(settings.precision)
    if mode == "second":
        trainer = checkpoint.load_training(directory / "checkpoint", device)
        if damage == ["return_scale"]:
            with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                fresh = Trainer(settings, device).ppo.return_scale
            trainer.ppo.return_scale.low, trainer.ppo.return_scale.high = fresh.low, fresh.high
        checkpoint.restore_default_generators(directory / "checkpoint", device)
        if damage == ["league"]:
            noise = torch.Generator().manual_seed(1)
            with torch.no_grad():
                for parameter in trainer.league.parameters():
                    parameter.add_(0.1 * torch.randn(parameter.shape, generator=noise).to(device))
    else:
        trainer = Trainer(settings, device)
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
