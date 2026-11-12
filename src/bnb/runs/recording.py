"""One update's record (diagnostics spec 1-3): the trainer's series, with what only the
running process knows (the time breakdown, the throughput, the memory, the compiled graphs,
the run's hours), the trigger rules' flags, written to the run's sinks.

The record adds, per update:
- `train/time/{rollout, learner, checkpoint, logging, update}_seconds`: the update's phases;
  `logging` is the previous record's writing, `update` the wall clock between two records;
- `train/environment_steps_per_second`, `train/ship_tokens_per_second`;
- `train/memory_peak_gigabytes` (the allocator's peak over the update) and
  `train/memory_budget_gigabytes` (the device's memory less the probe's margin), on a GPU;
- `counters/hours`, `counters/skipped_steps` (the run's total), `counters/recompiles` (graphs
  compiled after the segment's first update);
- in a chain's stage (`bnb.runs.chain`), `counters/chain_environment_steps`: the update's
  environment steps on the chain's axis (the stage's own plus the chain's before it);
- the trigger rules' `alerts/*`.
"""

import pathlib
import time

import torch

from . import probe
from .records import Run
from .sinks import Sinks
from .triggers import Triggers

DUMPS = "dumps"


def compiled_graphs() -> int:
    """The graphs the compiler has produced in this process."""
    from torch._dynamo.utils import counters

    return int(counters["stats"]["unique_graphs"])


class Recorder:
    """Builds and writes the records of one process's segment of `run`: `trainer`'s rows,
    the rule state of `triggers`, written to `sinks` (None: build only)."""

    def __init__(self, run: Run | None, trainer, sinks: Sinks | None, triggers: Triggers) -> None:
        self.run, self.trainer, self.sinks, self.triggers = run, trainer, sinks, triggers
        trainer.triggers = triggers  # their state goes into the training checkpoint
        self.device = trainer.rollout.device
        self.started = self.previous = time.perf_counter()
        self.hours = float(run.record.get("hours", 0.0)) if run is not None else 0.0
        self.skipped = int(run.record.get("skipped_steps", 0)) if run is not None else 0
        self.baseline: int | None = None
        self.logging_seconds = 0.0
        self.checkpoint_seconds = 0.0

    def record(self, row: dict) -> dict:
        """The record of the update `row` (`bnb.runtime.train.Trainer.update`), with the
        rules' flags, written to the sinks."""
        now = time.perf_counter()
        trainer = self.trainer
        record = dict(row)
        update_seconds = now - self.previous
        self.hours += update_seconds / 3600
        self.previous = now
        self.skipped += int(row.get("train/skipped_steps", 0))
        graphs = compiled_graphs()
        if self.baseline is None:
            self.baseline = graphs
        steps = trainer.steps_per_update
        record.update(
            {
                "train/time/checkpoint_seconds": self.checkpoint_seconds,
                "train/time/logging_seconds": self.logging_seconds,
                "train/time/update_seconds": update_seconds,
                "train/environment_steps_per_second": steps / update_seconds,
                "train/ship_tokens_per_second": steps
                * trainer.config.game.num_ships
                / update_seconds,
                "counters/hours": self.hours,
                "counters/skipped_steps": self.skipped,
                "counters/recompiles": graphs - self.baseline,
            }
        )
        chain = self.run.record.get("chain") if self.run is not None else None
        if chain is not None:
            record["counters/chain_environment_steps"] = (
                chain["start_step"] + row["counters/environment_steps"]
            )
        if self.device.type == "cuda":
            peaks = probe.measure(self.device)
            record["train/memory_peak_gigabytes"] = peaks["reserved"] / 2**30
            record["train/memory_budget_gigabytes"] = peaks["budget"] / 2**30
            probe.begin(self.device)
        self.checkpoint_seconds = 0.0
        record.update(self.triggers.check(record, row["counters/updates"]))
        if self.sinks is not None:
            self.sinks.write(record)
        if self.run is not None:
            self.run.record.update(hours=self.hours, skipped_steps=self.skipped)
        self.logging_seconds = time.perf_counter() - now
        return record


def dump_batch(run: Run, trainer, record: dict) -> pathlib.Path:
    """The skipped-step dump (decisions 4d): the update's whole batch as the segment store
    holds it (every stored field of every view, and the shot labels), with the update's
    record, under `dumps/step_<environment steps>/` in the run. Returns the directory."""
    import json

    store = trainer.rollout.store
    directory = run.directory / DUMPS / f"step_{trainer.ppo.environment_steps}"
    directory.mkdir(parents=True, exist_ok=True)
    store.synchronize()
    torch.save(
        {
            name: [leaf.to("cpu", copy=True) for leaf in leaves]
            for name, leaves in store.data.items()
        },
        directory / "batch.pt",
    )
    numbers = {
        name: value for name, value in record.items() if isinstance(value, int | float | bool)
    }
    (directory / "record.json").write_text(json.dumps(numbers, indent=1) + "\n")
    return directory
