"""One process's segment of a run (runs spec 5 and 7): the probe on its first update,
checkpoints every N environment steps with the retention, and the run's status at the end.
A BC run (`clone`) instead saves its clone as the run's policy file when a stop rule fires.

Ctrl-C finishes the current update, writes a checkpoint and marks the run stopped; an
exception marks it failed, with the message, before it propagates. The cadence N, when the
run does not set it, is measured: about 30 minutes at the throughput of the segment's
second update (the first compiles), in whole updates, recorded in run.json.
"""

import signal
import time
from collections.abc import Callable

from bnb.learn.bc import StopConfig
from bnb.runtime import cloning
from bnb.runtime.train import Trainer

from . import checkpoint, probe
from .records import Run

CADENCE_SECONDS = 30 * 60


def save(run: Run, trainer: Trainer, final: bool = False) -> None:
    """A training checkpoint at the trainer's step (unless one is there), then the
    retention."""
    step = trainer.ppo.env_steps
    path = run.checkpoint_path(step)
    if not path.exists():
        checkpoint.save_training(path, trainer, extra=dict(run=run.record["name"]))
    run.added_checkpoint(step, final=final)


def save_clone(run: Run, trainer: Trainer) -> None:
    """The clone as the run's policy file: its weights, the return scale and the step counts,
    for the BC -> RL hand-off (`bnb train --init-from`)."""
    checkpoint.save_policy(run.policy_path, trainer.model, return_scale=trainer.ppo.return_scale,
                           steps=dict(environment_steps=trainer.ppo.env_steps,
                                      updates=trainer.updates_done),
                           game=trainer.config.game, types=trainer.ppo.types)


def clone(run: Run, trainer: Trainer, stop: StopConfig = StopConfig(),
          log: Callable[[dict], None] = print) -> str:
    """BC until a stop rule fires ("finished") or Ctrl-C ("stopped", after the current
    update); either way the clone is saved as the run's policy file, and the rule (or
    "interrupted") is recorded as the run's `stopped_by`."""
    stopping = []
    previous = signal.getsignal(signal.SIGINT)

    def interrupt(number, frame) -> None:
        if stopping:  # a second Ctrl-C stops at once
            raise KeyboardInterrupt
        stopping.append(True)
        print("stopping after this update (Ctrl-C again to stop now)", flush=True)

    signal.signal(signal.SIGINT, interrupt)
    try:
        reason = cloning.clone(trainer, stop, log, interrupted=lambda: bool(stopping))
        save_clone(run, trainer)
        status = "stopped" if reason == "interrupted" else "finished"
        run.update(stopped_by=reason, policy=str(run.policy_path))
        run.end_segment(trainer.ppo.env_steps, status)
        return status
    except BaseException as error:
        run.end_segment(trainer.ppo.env_steps, "failed",
                        message=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.signal(signal.SIGINT, previous)


def train(run: Run, trainer: Trainer, log: Callable[[dict], None] = print,
          margin: float = probe.MARGIN) -> str:
    """Train until the update budget (`trainer.config.updates`, the run's total) or Ctrl-C;
    returns the run's final status ("finished" or "stopped")."""
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
        first = True
        while trainer.updates_done < trainer.config.updates and not stopping:
            if first:
                probe.begin(device)
            started = time.perf_counter()
            before = trainer.ppo.env_steps
            row = trainer.update()
            if first:
                peaks = probe.measure(device, margin)
                run.update(probe=peaks)
                print(probe.describe(peaks), flush=True)
                learner = trainer.config.learner
                probe.check(peaks, f"{trainer.config.num_envs} environments, "
                                   f"{learner.micro_batch_rows} micro-batch rows")
            elif run.record["checkpoint_every"] is None:
                seconds = time.perf_counter() - started
                updates = max(1, round(CADENCE_SECONDS / seconds))
                run.update(checkpoint_every=updates * trainer.steps_per_update)
            first = False
            log(row)
            every = run.record["checkpoint_every"]
            if every and trainer.ppo.env_steps // every > before // every:
                save(run, trainer)
        status = "stopped" if stopping else "finished"
        save(run, trainer, final=status == "finished")
        run.end_segment(trainer.ppo.env_steps, status)
        return status
    except BaseException as error:
        run.end_segment(trainer.ppo.env_steps, "failed",
                        message=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.signal(signal.SIGINT, previous)
