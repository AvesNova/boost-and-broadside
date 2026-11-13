"""One process's segment of a run (runs spec 5 and 7): the probe on its first update,
checkpoints every N environment steps with the retention (the newest two and the final),
the policy files of every update (`policies/`, when the run keeps them), and the run's
status at the end.
A PPO run trains to its update budget; a BC run until its stop rule fires, and then (or at
Ctrl-C) also saves its policy files: the average its score stop read as the run's policy,
and the live clone and both averages beside it.

A stop request (`StopRequest`: SIGINT, as Ctrl-C sends, SIGTERM, as `systemctl stop` sends,
or a `STOP` file in the run directory, checked after each update) finishes the current
update, writes a checkpoint, records the request in run.json (`stop_request`: what asked
and the step) and marks the run stopped; a second signal aborts at once. An exception marks
it failed, with the message, before it propagates. The cadence N, when the
run does not set it, is measured: about 30 minutes at the throughput of the segment's
second update (the first compiles), in whole updates, recorded in run.json.
"""

import copy
import pathlib
import shutil
import signal
import time
from collections.abc import Callable

import torch

from bnb.runtime import cloning
from bnb.runtime.lanes import EXPONENTIAL_AVERAGE, LIVE, UNIFORM_AVERAGE
from bnb.runtime.train import Trainer

from . import checkpoint, probe, recording
from .recording import Recorder
from .records import Run

CADENCE_SECONDS = 30 * 60
# the updates after the first that measure the cadence, by the slowest: one PPO update that
# stops early on its KL is several times shorter than the rest (2026-11-06: an 80-minute
# cadence measured on a 99-second update)
CADENCE_UPDATES = 3
# a BC run's policy files beside `policy/`, `policy-<variant>/`, by the lane player they hold
POLICY_VARIANTS = {LIVE: "live", UNIFORM_AVERAGE: "uniform", EXPONENTIAL_AVERAGE: "exponential"}
# the BC -> RL hand-off, a BC run's `policy/`: always the exponential average, for every
# model, never chosen by the score stop's windows (the author, 2026-11-11 and 2026-11-12)
HANDOFF_PLAYER = EXPONENTIAL_AVERAGE
# the file in a run directory that asks its process to stop after the update in progress
STOP_FILE = "STOP"
BY_STOP_FILE = "STOP file"  # a `StopRequest.by`, beside the signals' names
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM)


class StopRequest:
    """A request to stop after the update in progress, by SIGINT or SIGTERM (while entered:
    the handlers are this process's until exit) or by a run's STOP file (`check`). `by` is
    what asked first ("SIGINT", "SIGTERM" or "STOP file"), None before; a second signal
    raises KeyboardInterrupt, stopping at once. The entry point enters it before it builds
    anything, so a signal during the setup stops the run after no update."""

    def __init__(self) -> None:
        self.by: str | None = None
        self.previous: dict = {}

    def __enter__(self) -> StopRequest:
        for number in STOP_SIGNALS:
            self.previous[number] = signal.signal(number, self._handle)
        return self

    def __exit__(self, *exception) -> None:
        for number, handler in self.previous.items():
            signal.signal(number, handler)
        self.previous.clear()

    def _handle(self, number: int, frame) -> None:
        if self.by is not None and self.by != BY_STOP_FILE:  # a second signal stops at once
            raise KeyboardInterrupt
        self.by = signal.Signals(number).name
        print(f"{self.by}: stopping after this update (again to stop now)", flush=True)

    def check(self, directory: pathlib.Path) -> bool:
        """Whether a stop was asked for, reading `directory`'s STOP file."""
        if self.by is None and (directory / STOP_FILE).exists():
            self.by = BY_STOP_FILE
            print(f"{directory / STOP_FILE}: stopping after this update", flush=True)
        return self.by is not None


def save(run: Run, trainer: Trainer, final: bool = False, recorder: Recorder | None = None) -> None:
    """A training checkpoint at the trainer's step (unless one is there), then the
    retention; its time counts in the recorder's next record."""
    started = time.perf_counter()
    step = trainer.ppo.environment_steps
    path = run.checkpoint_path(step)
    if not path.exists():
        checkpoint.save_training(path, trainer, extra=dict(run=run.record["name"]))
    run.added_checkpoint(step, final=final)
    if recorder is not None:
        recorder.checkpoint_seconds += time.perf_counter() - started


@torch.no_grad()
def save_update_policies(
    run: Run, trainer: Trainer, scratch: torch.nn.Module, recorder: Recorder | None = None
) -> None:
    """The policy files of the update that just ended (after every update when the run keeps
    them, `update_policies`, else every `TrainConfig.policy_snapshot_every` environment
    steps and at each segment's end), at `policies/step_<steps>/`: the live
    weights (`live/`), the exponential average (`exponential_average/`) and, once it has
    started, the uniform average (`uniform_average/`). They are never pruned, so an average
    of another decay can be built afterwards from the live files (`bnb runs ema`). scratch: a
    host copy of the model the averages are loaded into. Files of the same step (a rewound or
    repeated update's) are replaced."""
    started = time.perf_counter()
    steps = trainer.ppo.environment_steps
    averages = getattr(trainer.league, "averages", None)  # a run with neither: the live only
    path = run.update_policies_path(steps)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir()

    def save(player: str, model: torch.nn.Module) -> None:
        checkpoint.save_policy(
            path / player,
            model,
            return_scale=trainer.ppo.return_scale,
            steps=dict(environment_steps=steps, updates=trainer.updates_done),
            game=trainer.config.game,
            types=trainer.ppo.types,
        )

    save(LIVE, trainer.model)
    # Each average is saved as soon as it is loaded into the one scratch model: saving after
    # loading both wrote the uniform average under both names (2026-11-10, every snapshot of
    # every run before this fix).
    for player in (EXPONENTIAL_AVERAGE, UNIFORM_AVERAGE) if averages is not None else ():
        if player == UNIFORM_AVERAGE and not averages.uniform_started:
            continue
        scratch.load_state_dict(averages.weights[player])
        save(player, scratch)
    if recorder is not None:
        recorder.checkpoint_seconds += time.perf_counter() - started


def act(run: Run, trainer: Trainer, recorder: Recorder, record: dict) -> None:
    """The skipped-step rule's action: a checkpoint and the batch dump, with the dump's
    path in run.json and on the console."""
    if not recorder.triggers.fired(record, "skipped_step"):
        return
    save(run, trainer, recorder=recorder)
    dumped = recording.dump_batch(run, trainer, record)
    run.update(dumps=run.record.get("dumps", []) + [str(dumped)])
    print(f"skipped a non-finite step: checkpoint and batch dump in {dumped}", flush=True)


def _replace_policy(
    path: pathlib.Path, trainer: Trainer, model: torch.nn.Module, ratings: dict | None
) -> None:
    """Write `model` as the policy file at `path` with the trainer's return scale and step
    counts and `ratings`, replacing an earlier one (a resumed run's)."""
    earlier = path.with_name(path.name + ".replaced")
    if path.exists():
        shutil.rmtree(earlier, ignore_errors=True)
        path.rename(earlier)
    checkpoint.save_policy(
        path,
        model,
        return_scale=trainer.ppo.return_scale,
        steps=dict(environment_steps=trainer.ppo.environment_steps, updates=trainer.updates_done),
        game=trainer.config.game,
        types=trainer.ppo.types,
        ratings=ratings,
    )
    shutil.rmtree(earlier, ignore_errors=True)


@torch.no_grad()
def save_policies(run: Run, trainer: Trainer) -> None:
    """A BC run's policy files, for the BC -> RL hand-off (`bnb train --init-from`): each of
    the live clone and the two averages of its weights (`bnb.league.lanes.LaneHooks`) at
    `policy-<variant>/` (POLICY_VARIANTS), and at `policy/` the hand-off, always the
    exponential average (HANDOFF_PLAYER), whatever the score stop reads. Each holds its
    weights, the return scale, the step counts and its own lane rating per mode (where a
    warm start's live rating starts). A resumed run's earlier files are replaced."""
    hooks = trainer.league  # BC's (`LaneHooks`): the lanes' ratings and the averages
    scratch = copy.deepcopy(trainer.model).cpu()
    for player, variant in POLICY_VARIANTS.items():
        if player == LIVE:
            model = trainer.model
        else:
            scratch.load_state_dict(hooks.averages.weights[player])
            model = scratch
        ratings = hooks.ratings.of(player)
        _replace_policy(run.variant_policy_path(variant), trainer, model, ratings)
        if player == HANDOFF_PLAYER:
            _replace_policy(run.policy_path, trainer, model, ratings)


def micro_batch_note(trainer: Trainer) -> str:
    """The learner's micro-batch sizes, which the compile of fixed shapes compiles every
    learner stage once each for: more than one when the minibatches do not divide the
    views (the cadence's halves of the first minibatch add their own)."""
    views = trainer.learner_pool.numel() * trainer.config.shards
    learner = trainer.ppo
    shapes = learner.micro_batch_shapes(views)
    cadence = sorted(set(learner.micro_batch_shapes(views, cadence=True)) - set(shapes))
    note = (
        f"learner micro-batches of {', '.join(map(str, shapes))} views "
        f"({views} views in {learner.config.minibatches} minibatches)"
    )
    if len(shapes) > 1:
        note += f": every compiled learner stage compiles {len(shapes)} times"
    if cadence:
        note += f"; the cadence adds {', '.join(map(str, cadence))}"
    return note


def train(
    run: Run,
    trainer: Trainer,
    log: Callable[[dict], None] = print,
    margin: float = probe.MARGIN,
    recorder: Recorder | None = None,
    stop: StopRequest | None = None,
) -> str:
    """Train until the budget or a stop request; returns the run's final status ("finished"
    or "stopped"). PPO's budget is the update count (`trainer.config.updates`, the run's
    total); BC trains until its stop rule fires (`bnb.runtime.cloning.update`), records the
    rule (or "interrupted") as the run's `stopped_by` and saves its policy files
    (`save_policies`), recording the player `policy/` holds (HANDOFF_PLAYER) as
    `policy_player`.
    recorder: builds and writes each update's record (`log` gets the record); without one
    `log` gets the trainer's row. stop: the entry point's entered `StopRequest` (else one is
    entered for the loop). A STOP file left by an earlier process is removed at the start;
    an honoured one after the stop's checkpoint."""
    if stop is None:
        with StopRequest() as own:
            return train(run, trainer, log, margin, recorder, own)
    cloned = trainer.config.objective == "bc"
    every_update = bool(run.record.get("update_policies"))
    snapshot_every = trainer.config.policy_snapshot_every
    scratch = copy.deepcopy(trainer.model).cpu()
    stop_file = run.directory / STOP_FILE
    if stop.by is None and stop_file.exists():
        print(f"removing {stop_file}, left by an earlier process", flush=True)
        stop_file.unlink()
    device = trainer.rollout.device
    if trainer.config.compiled:
        print(micro_batch_note(trainer), flush=True)
    try:
        first, reason, durations = True, None, []
        while not stop.by and reason is None:
            if not cloned and trainer.updates_done >= trainer.config.updates:
                break
            if first:
                probe.begin(device)
            started = time.perf_counter()
            before = trainer.ppo.environment_steps
            row, reason = cloning.update(trainer) if cloned else (trainer.update(), None)
            if first:
                peaks = probe.measure(device, margin)
                run.update(probe=peaks)
                print(probe.describe(peaks), flush=True)
                learner = trainer.config.learner
                probe.check(
                    peaks,
                    f"{trainer.config.environments} environments, "
                    f"{learner.micro_batch_rows} micro-batch rows",
                )
            elif run.record["checkpoint_every"] is None:
                durations.append(time.perf_counter() - started)
                if len(durations) == CADENCE_UPDATES:
                    updates = max(1, round(CADENCE_SECONDS / max(durations)))
                    run.update(checkpoint_every=updates * trainer.steps_per_update)
            first = False
            if recorder is not None:
                row = recorder.record(row)
                act(run, trainer, recorder, row)
            log(row)
            steps = trainer.ppo.environment_steps
            if every_update or steps // snapshot_every > before // snapshot_every:
                save_update_policies(run, trainer, scratch, recorder)
            every = run.record["checkpoint_every"]
            if every and trainer.ppo.environment_steps // every > before // every:
                save(run, trainer, recorder=recorder)
            stop.check(run.directory)
        step = trainer.ppo.environment_steps
        finished = reason is not None if cloned else trainer.updates_done >= trainer.config.updates
        status = "finished" if finished else "stopped"
        save(run, trainer, final=finished, recorder=recorder)
        if not run.update_policies_path(step).exists():  # the segment's end
            save_update_policies(run, trainer, scratch, recorder)
        if cloned:
            save_policies(run, trainer)
            run.update(
                stopped_by=reason or "interrupted",
                policy=str(run.policy_path),
                policy_player=HANDOFF_PLAYER,
            )
        if stop.by is not None and not finished:
            run.update(stop_request=dict(by=stop.by, step=step))
            if stop.by == BY_STOP_FILE:
                stop_file.unlink(missing_ok=True)
        run.end_segment(step, status)
        return status
    except BaseException as error:
        run.end_segment(
            trainer.ppo.environment_steps, "failed", message=f"{type(error).__name__}: {error}"
        )
        raise
