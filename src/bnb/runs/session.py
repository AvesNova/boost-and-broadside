"""One process's segment of a run (runs spec 5 and 7): the probe on its first update,
checkpoints every N environment steps with the retention, and the run's status at the end.
A PPO run trains to its update budget; a BC run until its stop rule fires, and then (or at
Ctrl-C) also saves its clone as the run's policy file.

Ctrl-C finishes the current update, writes a checkpoint and marks the run stopped; an
exception marks it failed, with the message, before it propagates. The cadence N, when the
run does not set it, is measured: about 30 minutes at the throughput of the segment's
second update (the first compiles), in whole updates, recorded in run.json.
"""

import shutil
import signal
import time
from collections.abc import Callable

from bnb.runtime import cloning
from bnb.runtime.train import Trainer

from . import checkpoint, probe, recording
from .recording import Recorder
from .records import Run

CADENCE_SECONDS = 30 * 60


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


def act(run: Run, trainer: Trainer, recorder: Recorder, record: dict) -> None:
    """The skipped-step rule's action: a checkpoint and the batch dump, with the dump's
    path in run.json and on the console."""
    if not recorder.triggers.fired(record, "skipped_step"):
        return
    save(run, trainer, recorder=recorder)
    dumped = recording.dump_batch(run, trainer, record)
    run.update(dumps=run.record.get("dumps", []) + [str(dumped)])
    print(f"skipped a non-finite step: checkpoint and batch dump in {dumped}", flush=True)


def save_clone(run: Run, trainer: Trainer) -> None:
    """The clone as the run's policy file: its weights, the return scale and the step counts,
    for the BC -> RL hand-off (`bnb train --init-from`). A resumed run's earlier clone is
    replaced."""
    path = run.policy_path
    earlier = path.with_name(path.name + ".replaced")
    if path.exists():
        shutil.rmtree(earlier, ignore_errors=True)
        path.rename(earlier)
    checkpoint.save_policy(
        path,
        trainer.model,
        return_scale=trainer.ppo.return_scale,
        steps=dict(environment_steps=trainer.ppo.environment_steps, updates=trainer.updates_done),
        game=trainer.config.game,
        types=trainer.ppo.types,
    )
    shutil.rmtree(earlier, ignore_errors=True)


def train(
    run: Run,
    trainer: Trainer,
    log: Callable[[dict], None] = print,
    margin: float = probe.MARGIN,
    recorder: Recorder | None = None,
) -> str:
    """Train until the budget or Ctrl-C; returns the run's final status ("finished" or
    "stopped"). PPO's budget is the update count (`trainer.config.updates`, the run's
    total); BC trains until its stop rule fires (`bnb.runtime.cloning.update`), records the
    rule (or "interrupted") as the run's `stopped_by` and saves the clone (`save_clone`).
    recorder: builds and writes each update's record (`log` gets the record); without one
    `log` gets the trainer's row."""
    cloned = trainer.config.objective == "bc"
    stopping = []
    previous = signal.getsignal(signal.SIGINT)

    def interrupt(number, frame) -> None:
        if stopping:  # a second Ctrl-C stops at once
            raise KeyboardInterrupt
        stopping.append(True)
        print("stopping after this update (Ctrl-C again to stop now)", flush=True)

    signal.signal(signal.SIGINT, interrupt)
    device = trainer.rollout.device
    try:
        first, reason = True, None
        while not stopping and reason is None:
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
                seconds = time.perf_counter() - started
                updates = max(1, round(CADENCE_SECONDS / seconds))
                run.update(checkpoint_every=updates * trainer.steps_per_update)
            first = False
            if recorder is not None:
                row = recorder.record(row)
                act(run, trainer, recorder, row)
            log(row)
            every = run.record["checkpoint_every"]
            if every and trainer.ppo.environment_steps // every > before // every:
                save(run, trainer, recorder=recorder)
        finished = reason is not None if cloned else not stopping
        status = "finished" if finished else "stopped"
        save(run, trainer, final=finished, recorder=recorder)
        if cloned:
            save_clone(run, trainer)
            run.update(stopped_by=reason or "interrupted", policy=str(run.policy_path))
        run.end_segment(trainer.ppo.environment_steps, status)
        return status
    except BaseException as error:
        run.end_segment(
            trainer.ppo.environment_steps, "failed", message=f"{type(error).__name__}: {error}"
        )
        raise
    finally:
        signal.signal(signal.SIGINT, previous)
