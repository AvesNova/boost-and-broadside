"""Where a run's series go (diagnostics spec 2): the run's local JSONL file, always, and W&B
when the run asks for it (`bnb train --wandb`). Both get the same records.

A record is one update's series by name (`bnb.runtime.metrics`): numbers, and tables as
JSON (`{"columns": [...], "rows": [[...]]}`). The x-axis of every W&B series is
`counters/environment_steps`. W&B's own settings (the entity, `WANDB_MODE=offline`) come
from its environment variables. Once the first record is logged, the W&B sink saves the
run's headline and curated views (`bnb.runtime.dashboards`) and prints their links; a
failure there, or `wandb-workspaces` missing (the `tools` extra), is a warning and the run
goes on. An offline run saves none: `bnb runs dashboard RUN` does once it is synced.
"""

import json
import pathlib
import secrets
import warnings

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


class DashboardWarning(UserWarning):
    """The run's W&B views could not be saved; the run goes on without them."""


def save_dashboards(name: str, entity: str, objective: str) -> dict[str, str] | None:
    """Save the run's views on W&B and print their links; on a failure (or without the
    `tools` extra) warn and return None."""
    try:
        from bnb.runtime import dashboards

        links = dashboards.save(name, entity, PROJECT, objective)
    except ImportError as error:
        warnings.warn(
            f"no dashboards: {error.name or error} is not installed (the tools extra, "
            f"uv sync --extra tools); `bnb runs dashboard {name}` saves them later",
            DashboardWarning,
            stacklevel=2,
        )
        return None
    except Exception as error:  # the run goes on: the views are a convenience
        warnings.warn(
            f"the dashboards were not saved ({type(error).__name__}: {error}); "
            f"`bnb runs dashboard {name}` tries again",
            DashboardWarning,
            stacklevel=2,
        )
        return None
    for view, link in links.items():
        print(f"dashboard {view}: {link}", flush=True)
    return links


def shutdown_telemetry() -> None:
    """Shut down W&B's own telemetry (wandb 0.31) before the process exits. Left to exit,
    its handler shuts the OpenTelemetry meter provider down, and then the provider's own
    exit handler, registered first so run last, shuts it down again: "shutdown can only be
    called once". Shut down here, the provider unregisters its handler and W&B's returns
    early. An offline run has no telemetry; a W&B without this module needs nothing."""
    try:
        from wandb.analytics.opentelemetry import opentelemetry_proxy
    except ImportError:
        return
    proxy = opentelemetry_proxy.get_open_telemetry_proxy()
    if proxy is not None:
        proxy.shutdown()


class WandbSink:
    """Streams each record to a W&B run named for the run, in project PROJECT, and saves the
    run's views after the first. A resumed run continues the W&B run whose id `run.json`
    keeps."""

    def __init__(self, run: Run, config: dict) -> None:
        import wandb

        identifier = run.record.get("wandb_id") or secrets.token_hex(4)  # W&B run ids: [a-z0-9]
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
        self.name, self.objective = run.record["name"], config.get("objective", "ppo")
        self.views_due = True

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
        if self.views_due:
            self.views_due = False
            self.save_views()

    def save_views(self) -> dict[str, str] | None:
        """Save the run's views and print their links (`save_dashboards`); an offline run
        saves none."""
        if self.session.offline:
            print(
                f"offline W&B run: no dashboards saved; `bnb runs dashboard {self.name}` "
                f"saves them once the run is synced",
                flush=True,
            )
            return None
        return save_dashboards(self.name, self.session.entity, self.objective)

    def close(self) -> None:
        """Finish the W&B run, then W&B's telemetry (`shutdown_telemetry`)."""
        self.session.finish()
        shutdown_telemetry()


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
