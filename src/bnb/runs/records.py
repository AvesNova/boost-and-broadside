"""The run directory and its record (runs spec 5).

    runs/<run>/
      run.json            the record: status, command, W&B id, segments, checkpoints
      config.json         the resolved TrainConfig at the start
      checkpoints/step_<environment steps>/   training checkpoints
      league/             the league's policy snapshots
      metrics.jsonl       the local metrics sink
      artifacts/<kind>/<id>/result.json        offline results

`run.json` is rewritten atomically on every change. Each process that runs the run adds a
segment: its commit, the steps and times it covered, and the run-time overrides it
applied.
"""

import datetime
import json
import os
import pathlib
import random
import shutil
import socket

from . import provenance
from .serialise import to_json

STATUSES = ("running", "stopped", "finished", "failed")
RECORD = "run.json"
CHECKPOINTS = "checkpoints"
RECENT = 8  # checkpoints kept newest-first, before the geometric history (runs spec 5)

# Small word lists for run names a person can remember (runs spec 5).
ADJECTIVES = ("amber", "brisk", "calm", "crimson", "dusty", "eager", "fuzzy", "gentle",
              "golden", "hollow", "icy", "jolly", "keen", "lively", "mellow", "misty",
              "nimble", "olive", "plucky", "purple", "quiet", "rapid", "rusty", "silver",
              "sleepy", "snowy", "sunny", "tidy", "vivid", "wandering", "witty", "zesty")
NOUNS = ("acorn", "badger", "beacon", "comet", "cricket", "falcon", "fern", "harbour",
         "heron", "lantern", "maple", "meadow", "otter", "pebble", "pumpkin", "quill",
         "raven", "river", "saffron", "sparrow", "teapot", "thistle", "tulip", "walnut",
         "willow", "yarrow", "kestrel", "marble", "puffin", "ember", "biscuit", "nebula")


def run_name(label: str, runs_directory: pathlib.Path, date: datetime.date | None = None,
             generator: random.Random | None = None) -> str:
    """`<YYYY-MM-DD>-<label>-<adjective>-<noun>`, drawn again until no run has it."""
    date = date or datetime.date.today()
    generator = generator or random.Random()
    for _ in range(1000):
        name = (f"{date.isoformat()}-{label}-{generator.choice(ADJECTIVES)}-"
                f"{generator.choice(NOUNS)}")
        if not (runs_directory / name).exists():
            return name
    raise RuntimeError(f"no free run name for {label} on {date} after 1000 draws")


def write_json(path: pathlib.Path, data) -> None:
    """Write JSON atomically: a flushed temporary file renamed over `path`."""
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w") as handle:
        handle.write(json.dumps(data, indent=1) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def process_alive(pid: int) -> bool:
    """Whether a process with this id exists on this host."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def retained(steps: list[int], every: int, final: int | None = None) -> set[int]:
    """The checkpoints (by step) the retention keeps: the newest RECENT, plus, of the older
    ones, the one nearest each age 8, 16, 32, ... checkpoints back (an age is the step gap
    to the newest over the cadence `every`), plus the final one. Every age beyond the oldest
    is nearest the oldest, so it is always kept."""
    if not steps:
        return set()
    ordered = sorted(steps)
    keep = set(ordered[-RECENT:])
    older, newest = ordered[:-RECENT], ordered[-1]
    if older:
        keep.add(older[0])
        age = RECENT
        while age < (newest - older[0]) / every:
            keep.add(min(older, key=lambda step: (abs((newest - step) / every - age), -step)))
            age *= 2
    if final is not None and final in steps:
        keep.add(final)
    return keep


class Run:
    """A run directory and its record (`record`, the parsed run.json)."""

    def __init__(self, directory: pathlib.Path, record: dict) -> None:
        self.directory, self.record = directory, record

    # ---- creating and opening

    @classmethod
    def create(cls, runs_directory: pathlib.Path, name: str, config, command: list[str],
               checkpoint_every: int, parent: dict | None = None) -> Run:
        """A new run directory with its config.json and run.json (status running, no
        segment yet). parent: the run and step a fork or a warm start began from."""
        directory = pathlib.Path(runs_directory) / name
        directory.mkdir(parents=True)
        for part in (CHECKPOINTS, "league", "artifacts"):
            (directory / part).mkdir()
        (directory / "metrics.jsonl").touch()
        write_json(directory / "config.json", to_json(config))
        record = dict(name=name, status="running", pid=os.getpid(), host=socket.gethostname(),
                      message=None, command=list(command), wandb_id=None, parent=parent,
                      checkpoint_every=checkpoint_every, newest_checkpoint=None,
                      final_checkpoint=None, probe=None, segments=[])
        run = cls(directory, record)
        run.save()
        return run

    @classmethod
    def open(cls, runs_directory: pathlib.Path, name: str) -> Run:
        """The run `name` under `runs_directory`."""
        directory = pathlib.Path(runs_directory) / name
        if not (directory / RECORD).is_file():
            raise FileNotFoundError(f"no run {name!r} in {runs_directory}")
        return cls(directory, json.loads((directory / RECORD).read_text()))

    def save(self) -> None:
        write_json(self.directory / RECORD, self.record)

    # ---- the record

    def start_segment(self, step: int, overrides: dict | None = None,
                      rewound_from: int | None = None) -> str | None:
        """Begin this process's segment at `step`, mark the run running, and return a
        warning when the commit differs from the previous segment's."""
        head, dirty = provenance.commit()
        segments = self.record["segments"]
        warning = None
        if segments and segments[-1]["commit"] != head:
            warning = (f"the run continues on commit {head}; its last segment ran on "
                       f"{segments[-1]['commit']}")
        segments.append(dict(commit=head, dirty=dirty, start_step=step,
                             start_time=provenance.now(), end_step=None, end_time=None,
                             overrides=dict(overrides or {}), rewound_from=rewound_from,
                             command=list(self.record["command"]) if not segments else None))
        self.record.update(status="running", pid=os.getpid(), host=socket.gethostname(),
                           message=None)
        self.save()
        return warning

    def end_segment(self, step: int, status: str, message: str | None = None) -> None:
        """End this process's segment at `step` with the run's new status."""
        if status not in STATUSES:
            raise ValueError(f"status {status!r} is not one of {STATUSES}")
        if self.record["segments"]:
            self.record["segments"][-1].update(end_step=step, end_time=provenance.now())
        self.record.update(status=status, message=message)
        self.save()

    def update(self, **fields) -> None:
        """Set fields of the record (the W&B id, the probe's peaks) and save it."""
        self.record.update(fields)
        self.save()

    def shown_status(self) -> str:
        """The status, with `running (stale)` for a running record whose process is gone
        from this host."""
        status = self.record["status"]
        if (status == "running" and self.record["host"] == socket.gethostname()
                and not process_alive(self.record["pid"])):
            return "running (stale)"
        return status

    # ---- checkpoints

    def checkpoint_path(self, step: int) -> pathlib.Path:
        return self.directory / CHECKPOINTS / f"step_{step}"

    def checkpoint_steps(self) -> list[int]:
        """The steps of the complete checkpoints, oldest first (a `.tmp` is not one)."""
        steps = []
        for path in (self.directory / CHECKPOINTS).iterdir():
            if path.name.startswith("step_") and not path.name.endswith(".tmp"):
                steps.append(int(path.name[len("step_"):]))
        return sorted(steps)

    def checkpoint_at(self, step: int | None = None) -> pathlib.Path:
        """The kept checkpoint at `step`, or the newest."""
        steps = self.checkpoint_steps()
        if not steps:
            raise FileNotFoundError(f"run {self.record['name']} has no checkpoint")
        if step is None:
            return self.checkpoint_path(steps[-1])
        if step not in steps:
            raise FileNotFoundError(f"run {self.record['name']} keeps no checkpoint at step "
                                    f"{step}; it keeps {steps}")
        return self.checkpoint_path(step)

    def set_aside_after(self, step: int) -> list[int]:
        """For a rewind to `step`: move every newer checkpoint into
        `checkpoints/rewound_<segment>/`, where it is kept and out of the retention, so the
        run can write those steps again. Returns their steps."""
        newer = [later for later in self.checkpoint_steps() if later > step]
        if newer:
            aside = self.directory / CHECKPOINTS / f"rewound_{len(self.record['segments'])}"
            aside.mkdir()
            for later in newer:
                self.checkpoint_path(later).rename(aside / f"step_{later}")
        if self.record["final_checkpoint"] in newer:
            self.record["final_checkpoint"] = None
        return newer

    def added_checkpoint(self, step: int, final: bool = False) -> None:
        """Record the checkpoint just written at `step` and apply the retention."""
        self.record["newest_checkpoint"] = step
        if final:
            self.record["final_checkpoint"] = step
        steps = self.checkpoint_steps()
        every = self.record["checkpoint_every"] or 1  # not measured yet: ages in steps
        keep = retained(steps, every, self.record["final_checkpoint"])
        for old in set(steps) - keep:
            shutil.rmtree(self.checkpoint_path(old))
        self.save()


def list_runs(runs_directory: pathlib.Path) -> list[Run]:
    """Every run under `runs_directory`, by name."""
    directory = pathlib.Path(runs_directory)
    if not directory.is_dir():
        return []
    return [Run.open(directory, path.name) for path in sorted(directory.iterdir())
            if (path / RECORD).is_file()]
