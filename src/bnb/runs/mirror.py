"""Mirror a run's local metrics into W&B (`bnb runs mirror RUN`): for a run trained without
`--wandb`, from a machine logged in to W&B, while it trains or after.

The mirror reads the run's `metrics.jsonl`, `config.json` and `run.json` (in a runs
directory, or fetched over ssh with rsync from `HOST:REMOTE_RUNS` into a local cache) and
logs each record as the live sink (`bnb.runs.sinks.WandbSink`) would have: project
`PROJECT`, the run's name, its config, `counters/environment_steps` as the x-axis, and a
chain's stages as two W&B runs grouped by the chain. RUN is a run or a chain (its stages
that exist).

It is idempotent. A run's W&B id is the one its run.json keeps (a run that streamed live),
else one derived from its name (`mirror_id`), and the W&B run is resumed; the newest step
mirrored is kept in `MIRROR_STATE` in the (local or cached) run directory and taken as the
larger of that and the W&B run's own summary, so a re-run or a poll logs only the records
after it. A record at or before that step (a run rewound by a resume) is left out: W&B
keeps one point per step. Only whole lines are read, so a line being written is left for
the next poll.

With `follow` it polls every `every` seconds until the run is over: the chain's (or run's)
exit file `<name>.exit` beside it in the runs directory (box/dry_run.sh writes one), or no
stage still running by its run.json; then one last sync.
"""

import hashlib
import json
import pathlib
import subprocess
import time
from collections.abc import Callable

from .chain import RECORD as CHAIN_RECORD
from .records import RECORD
from .sinks import METRICS, STEP, save_dashboards, shutdown_telemetry, wandb_group, wandb_values
from .upload import PROJECT

MIRROR_STATE = "wandb_mirror.json"  # {"wandb_id": ..., "step": newest step mirrored}
CONFIG = "config.json"
FETCHED = (RECORD, CONFIG, METRICS)  # a run's files the mirror reads


def mirror_id(name: str) -> str:
    """The W&B run id of a mirrored run named `name`: stable, so every mirror of it resumes
    the same W&B run."""
    return "m" + hashlib.sha256(name.encode()).hexdigest()[:15]


def whole_records(path: pathlib.Path) -> list[dict]:
    """The records of a metrics file's complete lines (a last line still being written, with
    no newline yet, is left out)."""
    if not path.is_file():
        return []
    text = path.read_text()
    return [json.loads(line) for line in text[: text.rfind("\n") + 1].splitlines() if line]


class Fetch:
    """Copies the files the mirror reads from `host:remote` into `cache` with rsync over ssh
    (only what changed; missing files are skipped)."""

    def __init__(self, host: str, remote: str, cache: pathlib.Path) -> None:
        self.host, self.remote, self.cache = host, remote.rstrip("/"), pathlib.Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)

    def __call__(self, paths: list[str]) -> None:
        """Fetch `paths`, relative to the remote runs directory."""
        subprocess.run(
            [
                "rsync",
                "-az",
                "--ignore-missing-args",
                "--files-from=-",
                f"{self.host}:{self.remote}/",
                f"{self.cache}/",
            ],
            input="\n".join(paths) + "\n",
            text=True,
            check=True,
            timeout=600,
        )


class Mirror:
    """Mirrors the run or chain `name` in `runs_directory` (the cache, with `fetch`)."""

    def __init__(
        self,
        runs_directory: pathlib.Path,
        name: str,
        fetch: Callable[[list[str]], None] | None = None,
        report: Callable[[str], None] = print,
    ) -> None:
        self.runs_directory, self.name = pathlib.Path(runs_directory), name
        self.fetch, self.report = fetch, report

    def chain(self) -> dict | None:
        """The chain's record, or None for a plain run."""
        path = self.runs_directory / self.name / CHAIN_RECORD
        return json.loads(path.read_text()) if path.is_file() else None

    def run_names(self) -> list[str]:
        """The runs to mirror: the chain's stages (BC first), or the run itself."""
        chain = self.chain()
        if chain is None:
            return [self.name]
        return [run for run in chain["stages"].values() if run]

    def refresh(self) -> None:
        """Fetch the chain's record, the exit file and every run's files (with `fetch`)."""
        if self.fetch is None:
            return
        self.fetch([f"{self.name}/{CHAIN_RECORD}", f"{self.name}.exit"])
        self.fetch(
            [f"{self.name}/{file}" for file in FETCHED]
            + [f"{run}/{file}" for run in self.run_names() for file in FETCHED]
        )

    def over(self) -> bool:
        """Whether the run is over: its exit file, or no run left to run (a chain's BC stage
        finished with no RL stage yet is not over)."""
        if (self.runs_directory / f"{self.name}.exit").exists():
            return True
        statuses = []
        for run in self.run_names():
            path = self.runs_directory / run / RECORD
            if path.is_file():
                statuses.append(json.loads(path.read_text())["status"])
        if not statuses or "running" in statuses:
            return False
        chain = self.chain()
        if chain is None:
            return True
        waiting = len(statuses) < len(self.run_names()) and statuses[-1] == "finished"
        return not waiting

    def found(self) -> bool:
        """Whether a run of the name exists (after `refresh`)."""
        return any((self.runs_directory / run / RECORD).is_file() for run in self.run_names())

    def sync(self) -> int:
        """Log every run's new records (after `refresh`); the number logged."""
        return sum(self.sync_run(self.runs_directory / run) for run in self.run_names())

    def sync_run(self, directory: pathlib.Path) -> int:
        """Log the run in `directory`'s records after its newest mirrored step; the number
        logged."""
        if not (directory / RECORD).is_file():
            return 0
        state_path = directory / MIRROR_STATE
        state = json.loads(state_path.read_text()) if state_path.is_file() else {}
        records = whole_records(directory / METRICS)
        newest = state.get("step")
        if records and newest is not None and records[-1].get(STEP, 0) <= newest:
            return 0
        if not records:
            return 0
        import wandb

        record = json.loads((directory / RECORD).read_text())
        config_path = directory / CONFIG
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        identifier = state.get("wandb_id") or record.get("wandb_id") or mirror_id(record["name"])
        session = wandb.init(
            project=PROJECT,
            name=record["name"],
            group=wandb_group(record),
            id=identifier,
            resume="allow",
            dir=str(directory),
            config=config,
        )
        logged = 0
        try:
            session.define_metric(STEP)
            session.define_metric("*", step_metric=STEP)
            summary_step = session.summary.get(STEP)
            candidates = [step for step in (newest, summary_step) if step is not None]
            newest = max(candidates) if candidates else None
            first = newest is None
            for entry in records:
                step = entry.get(STEP)
                if step is None or (newest is not None and step <= newest):
                    continue
                session.log(wandb_values(wandb, entry))
                newest, logged = step, logged + 1
            entity, offline = session.entity, session.offline
        finally:
            session.finish()
            state_path.write_text(json.dumps(dict(wandb_id=identifier, step=newest)) + "\n")
        self.report(f"{record['name']}: {logged} records mirrored, to step {newest}")
        if first and logged and not offline:
            save_dashboards(record["name"], entity, config.get("objective", "ppo"))
        return logged

    def follow(self, every: float) -> None:
        """Sync every `every` seconds until the run is over, then once more; a failed fetch
        is retried at the next poll."""
        while True:
            try:
                self.refresh()  # the exit file is fetched before the metrics, so the
            except (subprocess.SubprocessError, OSError) as error:  # the box unreachable
                self.report(f"fetch failed ({error}); trying again in {every:.0f} s")
                time.sleep(every)
                continue
            over = self.over()  # sync after seeing it has every record
            self.sync()
            if over:
                self.report(f"{self.name} is over: mirror done")
                return
            time.sleep(every)


def mirror(
    runs_directory: pathlib.Path | None,
    name: str,
    follow: bool = False,
    every: float = 300.0,
    host: str | None = None,
    remote_runs: str = "/workspace/runs",
    cache: pathlib.Path | None = None,
) -> None:
    """`bnb runs mirror`: mirror `name` once, or with `follow` until it is over; from
    `host:remote_runs` through `cache` (default ~/.cache/bnb/mirror/HOST) when `host` is
    given, else from `runs_directory`."""
    fetch = None
    if host is not None:
        cache = pathlib.Path(cache or pathlib.Path.home() / ".cache" / "bnb" / "mirror" / host)
        fetch, runs_directory = Fetch(host, remote_runs, cache), cache
    elif runs_directory is None:
        raise ValueError("mirror needs a runs directory or a host")
    session = Mirror(runs_directory, name, fetch, report=lambda line: print(line, flush=True))
    session.refresh()
    if not session.found():
        raise FileNotFoundError(f"no run or chain {name!r} in {host or runs_directory}")
    try:
        if follow:
            session.follow(every)
        else:
            session.sync()
    finally:
        shutdown_telemetry()
