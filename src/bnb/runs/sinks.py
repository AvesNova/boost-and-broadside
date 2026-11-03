"""Where a run's series go (diagnostics spec 2): the run's local JSONL file, always, and W&B
when the run asks for it (`bnb train --wandb`). Both get the same records.

A record is one update's series by name (`bnb.runtime.metrics`): numbers, and tables as
JSON (`{"columns": [...], "rows": [[...]]}`). The x-axis of every W&B series is
`counters/environment_steps`. W&B's own settings (the entity, `WANDB_MODE=offline`) come
from its environment variables.
"""

import json
import pathlib

from .records import Run
from .upload import PROJECT

STEP = "counters/environment_steps"  # every series' x-axis
METRICS = "metrics.jsonl"


def is_table(value) -> bool:
    """Whether a record's value is a table (`bnb.contracts.series.Table.as_json`)."""
    return isinstance(value, dict) and value.keys() == {"columns", "rows"}


class JsonSink:
    """Appends each record as one JSON line to `path` (written through on every record, so a
    crash loses nothing logged)."""

    def __init__(self, path: pathlib.Path) -> None:
        self.path = pathlib.Path(path)
        self.handle = open(self.path, "a")

    def write(self, record: dict) -> None:
        """Append `record`."""
        self.handle.write(json.dumps(record) + "\n")
        self.handle.flush()

    def close(self) -> None:
        """Close the file."""
        self.handle.close()


class WandbSink:
    """Streams each record to a W&B run named for the run, in project PROJECT. A resumed run
    continues the W&B run whose id `run.json` keeps."""

    def __init__(self, run: Run, config: dict) -> None:
        import wandb

        identifier = run.record.get("wandb_id") or wandb.util.generate_id()
        self.session = wandb.init(
            project=PROJECT,
            name=run.record["name"],
            id=identifier,
            resume="allow",
            dir=str(run.directory),
            config=config,
        )
        self.session.define_metric(STEP)
        self.session.define_metric("*", step_metric=STEP)
        if run.record.get("wandb_id") != identifier:
            run.update(wandb_id=identifier)
        self.wandb = wandb

    def write(self, record: dict) -> None:
        """Log `record`: numbers as series, tables as W&B tables; values that are neither
        (None, text) are left out."""
        logged = {}
        for name, value in record.items():
            if is_table(value):
                logged[name] = self.wandb.Table(columns=value["columns"], data=value["rows"])
            elif isinstance(value, bool | int | float):
                logged[name] = value
        self.session.log(logged)

    def close(self) -> None:
        """Finish the W&B run."""
        self.session.finish()


class Sinks:
    """Every sink of a run: the JSONL file, and W&B with `wandb`."""

    def __init__(self, run: Run, config: dict, wandb: bool = False) -> None:
        self.sinks: list = [JsonSink(run.directory / METRICS)]
        if wandb:
            self.sinks.append(WandbSink(run, config))

    def write(self, record: dict) -> None:
        """Write `record` to every sink."""
        for sink in self.sinks:
            sink.write(record)

    def close(self) -> None:
        """Close every sink."""
        for sink in self.sinks:
            sink.close()


def read_metrics(path: pathlib.Path) -> list[dict]:
    """A run's JSONL records, in order."""
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]
