"""One process of the exact-resume test (runs acceptance 3), as an entry point would run
it: seed the process, then train straight through, or train and save, or resume and train.

    python resume_worker.py MODE DIRECTORY DEVICE MODEL OBJECTIVE UPDATES [DAMAGE]

MODE: "straight" (2 x UPDATES updates), "first" (UPDATES updates, then a training
checkpoint at DIRECTORY/checkpoint) or "second" (resume from it, UPDATES more). Each writes
its metrics and its final weights into DIRECTORY/<MODE>. The league (`bnb.league`) is
attached, with a snapshot every update, so the pool, the gauges, the slots and the drains
all change before the checkpoint; its snapshots go to DIRECTORY/league (DIRECTORY/straight-
league for the straight run). DAMAGE (with "second", to show the test sees it): "league"
adds noise to the opponent slots' weights after the restore, "draws" restarts the league's
slot draws, "gauges" forgets the gauges' records and windows, "lanes" forgets the rating
lanes' records, "return_scale" puts back a fresh trainer's return scale, "stop_rule" forgets
BC's stop rule's window. Episodes are short (2 s), so drains end and begin in each half.
An update is two shards and a lookahead, beside a few rating lanes. OBJECTIVE "bc" clones
instead (no league; its lanes' ratings are the trainer's hooks): each update feeds the stop
rule (`bnb.runtime.cloning.update`), whose window of lane games has filled and trims by the
checkpoint.
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
from bnb.league.lanes import LaneHooks  # noqa: E402
from bnb.league.league import League  # noqa: E402
from bnb.learn import bc  # noqa: E402
from bnb.learn.ppo import LearnerConfig  # noqa: E402
from bnb.models.relational import RelationalConfig  # noqa: E402
from bnb.runs import checkpoint  # noqa: E402
from bnb.runs.triggers import Triggers  # noqa: E402
from bnb.runtime import cloning  # noqa: E402
from bnb.runtime.lanes import BC_MATCHUPS, LIVE, SCRIPTED_PLAYER, LaneConfig, Matchup  # noqa: E402
from bnb.runtime.opponents import LeagueConfig  # noqa: E402
from bnb.runtime.train import TrainConfig, Trainer  # noqa: E402

TIMES = (
    "train/time/rollout_seconds",
    "train/time/learner_seconds",
    "train/time/cadence_seconds",
)
ENVIRONMENTS, LENGTH = 16, 32
# The geometry the resume must carry: consecutive shards, a lookahead after them (the carry
# saved is the state after it) and rating lanes of every kind of player.
GEOMETRY = dict(shards=2, lookahead=LENGTH)
RL_LANES = LaneConfig(
    environments=6,
    matchups=(
        Matchup(LIVE, SCRIPTED_PLAYER),
        Matchup("exponential_average", SCRIPTED_PLAYER),
        Matchup("rung_0.9", SCRIPTED_PLAYER),
    ),
    window=2,
)
BC_LANES = LaneConfig(environments=4, matchups=BC_MATCHUPS, window=2)


def config(model: str, device: str, objective: str = "ppo") -> TrainConfig:
    # A learning rate that moves the model visibly in two updates, and a return-scale floor
    # under the returns' spread, so both the league's snapshots and the return scale show.
    settings = dict(
        epochs=2,
        minibatches=2,
        learning_rate=3e-3,
        warmup_steps=1000,
        micro_batch_rows=64,
        return_scale_floor=1e-7,
    )
    if objective == "bc":  # a window of a few games, so it fills and trims in each half
        return TrainConfig(
            game=GameConfig(2, 2, 1500.0, episode_seconds=0.5, respawn_seconds=0.2),
            pools=cloning.LINEUP,
            environments=ENVIRONMENTS,
            updates=0,
            length=LENGTH,
            **GEOMETRY,
            lanes=BC_LANES,
            model=model,
            relational=RelationalConfig(hidden=32),
            learner=bc.config(**settings),
            compiled=device == "cuda",
            objective="bc",
            stop=bc.StopConfig(score_window=4, plateau_steps=ENVIRONMENTS * LENGTH),
        )
    learner = LearnerConfig(**settings)
    return TrainConfig(
        game=GameConfig(2, 2, 1500.0, episode_seconds=2.0, respawn_seconds=1.0),
        environments=ENVIRONMENTS,
        updates=0,
        length=LENGTH,
        **GEOMETRY,
        lanes=RL_LANES,
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
    mode, directory, device, model, objective, updates, *damage = sys.argv[1:]
    directory, device, updates = pathlib.Path(directory), torch.device(device), int(updates)
    settings = config(model, device.type, objective)
    cloned = objective == "bc"
    torch.manual_seed(settings.seed)
    set_precision(settings.precision)
    if mode == "second":
        trainer = checkpoint.load_training(directory / "checkpoint", device)
        if damage == ["return_scale"]:
            with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                fresh = Trainer(settings, device).ppo.return_scale
            trainer.ppo.return_scale.low, trainer.ppo.return_scale.high = fresh.low, fresh.high
        checkpoint.restore_default_generators(directory / "checkpoint", device)
        trainer.triggers = Triggers(
            state=checkpoint.triggers_state(directory / "checkpoint"), objective=objective
        )
        if not cloned:
            league = League(
                trainer,
                directory / "league",
                state=checkpoint.league_state(directory / "checkpoint"),
            )
        else:
            league = LaneHooks(trainer, checkpoint.league_state(directory / "checkpoint"))
        dump_state(directory / "second", trainer)  # the restore alone, before any damage
        if damage == ["stop_rule"]:
            trainer.stop_rule = bc.StopRule(settings.stop)
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
        if damage == ["lanes"]:
            lanes = league.ratings if cloned else league.lanes
            lanes.whole.clear()
            lanes.recent.clear()
        if damage == ["gauges"]:
            for gauge in league.gauges.values():
                gauge.wins.clear()
                for window in gauge.windows.values():
                    window.clear()
    else:
        trainer = Trainer(settings, device)
        trainer.triggers = Triggers(objective=objective)
        if not cloned:
            League(trainer, directory / ("straight-league" if mode == "straight" else "league"))
        else:
            LaneHooks(trainer)
    count = 2 * updates if mode == "straight" else updates
    rows = []
    for _ in range(count):
        row = cloning.update(trainer)[0] if cloned else trainer.update()
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
