"""One process of the exact-resume test (runs acceptance 3), as an entry point would run
it: seed the process, then train straight through, or train and save, or resume and train.

    python resume_worker.py MODE DIRECTORY DEVICE MODEL UPDATES [DAMAGE]

MODE: "straight" (2 x UPDATES updates), "first" (UPDATES updates, then a training
checkpoint at DIRECTORY/checkpoint) or "second" (resume from it, UPDATES more). Each writes
its metrics and its final weights into DIRECTORY/<MODE>. The league (`bnb.league`) is
attached, with a snapshot every update, so the pool, the gauges, the slots and the drains
all change before the checkpoint; its snapshots go to DIRECTORY/league (DIRECTORY/straight-
league for the straight run). DAMAGE (with "second", to show the test sees it): "league"
adds noise to the opponent slots' weights after the restore, "draws" restarts the league's
slot draws, "gauges" forgets the gauges' records and windows, "return_scale" puts back a fresh
trainer's return scale. Episodes are short (2 s), so drains end and begin in each half.
"""

import json
import os
import pathlib
import sys

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")  # as the entry points set it

import safetensors.torch  # noqa: E402
import torch  # noqa: E402

from bnb.cli import set_precision  # noqa: E402
from bnb.contracts import GameConfig  # noqa: E402
from bnb.league.league import League  # noqa: E402
from bnb.learn.ppo import LearnerConfig  # noqa: E402
from bnb.models.relational import RelationalConfig  # noqa: E402
from bnb.runs import checkpoint  # noqa: E402
from bnb.runs.triggers import Triggers  # noqa: E402
from bnb.runtime.opponents import LeagueConfig  # noqa: E402
from bnb.runtime.train import TrainConfig, Trainer  # noqa: E402

TIMES = (
    "train/time/rollout_seconds",
    "train/time/learner_seconds",
    "train/time/cadence_seconds",
)
ENVIRONMENTS, LENGTH = 16, 32


def config(model: str, device: str) -> TrainConfig:
    # A learning rate that moves the model visibly in two updates, and a return-scale floor
    # under the returns' spread, so both the league's snapshots and the return scale show.
    learner = LearnerConfig(
        epochs=2,
        minibatches=2,
        learning_rate=3e-3,
        warmup_steps=1000,
        micro_batch_rows=64,
        return_scale_floor=1e-7,
    )
    return TrainConfig(
        game=GameConfig(2, 2, 1500.0, episode_seconds=2.0, respawn_seconds=1.0),
        environments=ENVIRONMENTS,
        updates=0,
        length=LENGTH,
        model=model,
        relational=RelationalConfig(hidden=32),
        league=LeagueConfig(snapshot_every=ENVIRONMENTS * LENGTH),
        learner=learner,
        compiled=device == "cuda",
    )


def dump_state(out: pathlib.Path, trainer: Trainer) -> None:
    """The whole training state as it is now, listed by the checkpoint's own enumeration
    (`checkpoint.training_state`), for the test to compare bit for bit: the first process
    dumps it as it saves, the second as soon as it has restored, before any compute."""
    out.mkdir(parents=True, exist_ok=True)
    record, tensors = checkpoint.training_state(trainer)
    (out / "state.json").write_text(json.dumps(record, sort_keys=True))
    torch.save(tensors, out / "state.pt")


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
        trainer.triggers = Triggers(state=checkpoint.triggers_state(directory / "checkpoint"))
        league = League(
            trainer, directory / "league", state=checkpoint.league_state(directory / "checkpoint")
        )
        dump_state(directory / "second", trainer)  # the restore alone, before any damage
        if damage == ["league"]:
            noise = torch.Generator().manual_seed(1)
            with torch.no_grad():
                for parameter in trainer.opponents.parameters.values():
                    if parameter.is_floating_point():
                        parameter.add_(
                            0.1 * torch.randn(parameter.shape, generator=noise).to(device)
                        )
        if damage == ["ulp"]:
            with torch.no_grad():  # every weight one step toward +infinity
                for parameter in trainer.model.parameters():
                    if parameter.is_floating_point():
                        parameter.copy_(torch.nextafter(parameter, torch.full_like(parameter, 1e9)))
        if damage == ["order_seed"]:
            trainer.ppo.order_generator.manual_seed(12345)
        if damage == ["draws"]:
            league.swaps = 0
        if damage == ["gauges"]:
            for gauge in league.gauges.values():
                gauge.wins.clear()
                for window in gauge.windows.values():
                    window.clear()
    else:
        trainer = Trainer(settings, device)
        trainer.triggers = Triggers()
        League(trainer, directory / ("straight-league" if mode == "straight" else "league"))
    count = 2 * updates if mode == "straight" else updates
    rows = []
    for _ in range(count):
        row = trainer.update()
        row.update(trainer.triggers.check(row, row["counters/updates"]))
        rows.append({name: value for name, value in row.items() if name not in TIMES})
    if mode == "first":
        checkpoint.save_training(directory / "checkpoint", trainer)
        dump_state(directory / "first", trainer)
    out = directory / mode
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(rows))
    safetensors.torch.save_file(checkpoint.weights(trainer.model), out / "weights.safetensors")


if __name__ == "__main__":
    main()
