"""An exponential moving average of any decay, built offline from a run's saved live
policies (`policies/step_<steps>/live/`, one per update; see `bnb.runs.session`), written
as a policy file the tournament can rate.

The average folds the live weights update by update, as the run's own does
(`bnb.league.pool.average_into`): `average = decay * average + (1 - decay) * live`, so the
decay is per update, not per environment step. The run's own average starts from the
model's initial weights; this one starts from the first saved update's live weights, so
for a decay near 1 (a horizon of many updates) its earliest part is weighted slightly
differently. Integer buffers take the newest value. The average ends at `--at` (default:
the newest saved update), and the return scale is the end update's.
"""

import pathlib

import safetensors.torch
import torch

from . import provenance
from .checkpoint import RETURN_SCALE, WEIGHTS, read_policy, write_checkpoint
from .records import POLICIES, Run, step_of

LIVE = "live"


def live_steps(run: Run) -> list[int]:
    """The environment steps of the run's saved live policies, oldest first."""
    directory = run.directory / POLICIES
    steps = []
    if directory.is_dir():
        for path in directory.iterdir():
            step = step_of(path)
            if step is not None and (path / LIVE).is_dir():
                steps.append(step)
    return sorted(steps)


def average(run: Run, decay: float, at: int | None = None) -> tuple[dict, int, int]:
    """(the average's weights, the step it ends at, the updates folded in) over the saved
    live policies up to `at`."""
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"decay {decay}: need 0 <= decay < 1")
    steps = [step for step in live_steps(run) if at is None or step <= at]
    if not steps:
        raise FileNotFoundError(
            f"run {run.record['name']} has no saved live policies"
            + ("" if at is None else f" up to step {at}")
            + " (it was run without per-update policies?)"
        )
    weights: dict[str, torch.Tensor] = {}
    for step in steps:
        live = safetensors.torch.load_file(run.update_policies_path(step) / LIVE / WEIGHTS)
        for name, value in live.items():
            if name not in weights or not value.is_floating_point():
                weights[name] = value.clone()
            else:
                weights[name].mul_(decay).add_(value, alpha=1.0 - decay)
    return weights, steps[-1], len(steps)


def build(run: Run, decay: float, at: int | None = None, out: str | None = None) -> pathlib.Path:
    """Build the run's EMA of `decay` ending at step `at` and write it as a policy file at
    `out` (default: `<run>/artifacts/ema/decay_<decay>_step_<steps>/`); returns the path."""
    weights, steps, updates = average(run, decay, at)
    last = run.update_policies_path(steps) / LIVE
    record = dict(read_policy(last).raw)
    record.pop("files")
    record.update(
        steps=dict(record["steps"], ema_decay=decay, ema_updates=updates),
        time=provenance.now(),
    )
    files = {WEIGHTS: weights}
    if record["return_scale"]:
        files[RETURN_SCALE] = safetensors.torch.load_file(last / RETURN_SCALE)
    path = (
        pathlib.Path(out)
        if out is not None
        else run.directory / "artifacts" / "ema" / f"decay_{decay}_step_{steps}"
    )
    return write_checkpoint(path, record, files)
