"""The run directory and its record (runs spec 5).

    runs/<run>/
      run.json            the record: status, command, W&B id, segments, checkpoints
      config.json         the resolved TrainConfig at the start
      checkpoints/step_<environment steps>/   training checkpoints
      league/             the league's policy snapshots
      policy/             a BC run's clone, a policy file (`bnb train --objective bc`)
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
from collections.abc import Iterable

from . import provenance
from .serialise import to_json

STATUSES = ("running", "stopped", "finished", "failed")
RECORD = "run.json"
CHECKPOINTS = "checkpoints"
POLICIES = "policies"  # every update's policy files (update_policies)
RECENT = 2  # training checkpoints kept, newest first (and the final one)

# Small word lists for run names a person can remember (runs spec 5).
ADJECTIVES = (
    "amber",
    "brisk",
    "calm",
    "crimson",
    "dusty",
    "eager",
    "fuzzy",
    "gentle",
    "golden",
    "hollow",
    "icy",
    "jolly",
    "keen",
    "lively",
    "mellow",
    "misty",
    "nimble",
    "olive",
    "plucky",
    "purple",
    "quiet",
    "rapid",
    "rusty",
    "silver",
    "sleepy",
    "snowy",
    "sunny",
    "tidy",
    "vivid",
    "wandering",
    "witty",
    "zesty",
)
NOUNS = (
    "acorn",
    "badger",
    "beacon",
    "comet",
    "cricket",
    "falcon",
    "fern",
    "harbour",
    "heron",
    "lantern",
    "maple",
    "meadow",
    "otter",
    "pebble",
    "pumpkin",
    "quill",
    "raven",
    "river",
    "saffron",
    "sparrow",
    "teapot",
    "thistle",
    "tulip",
    "walnut",
    "willow",
    "yarrow",
    "kestrel",
    "marble",
    "puffin",
    "ember",
    "biscuit",
    "nebula",
)


def run_name(
    label: str,
    runs_directory: pathlib.Path,
    date: datetime.date | None = None,
    generator: random.Random | None = None,
) -> str:
    """`<YYYY-MM-DD>-<label>-<adjective>-<noun>`, drawn again until no run has it."""
    date = date or datetime.date.today()
    generator = generator or random.Random()
    for _ in range(1000):
        name = (
            f"{date.isoformat()}-{label}-{generator.choice(ADJECTIVES)}-{generator.choice(NOUNS)}"
        )
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


def retained(ordinals: Iterable[int], final: int | None = None) -> set[int]:
    """The training checkpoints the retention keeps, by ordinal (0, 1, 2, ... in the order
    written, a stop's off-cadence checkpoint included): the newest RECENT, and the final one.
    The small policy files of every update (`policies/`) are the run's history; the large
    resumable states are kept only to resume from. The league's snapshots have their own rule
    (`bnb.league.pool.retained`)."""
    ordinals = set(ordinals)
    if not ordinals:
        return set()
    newest = max(ordinals)
    keep = {ordinal for ordinal in ordinals if newest - ordinal < RECENT}
    if final in ordinals:
        keep.add(final)
    return keep


def step_of(path: pathlib.Path) -> int | None:
    """The step of a complete checkpoint directory `step_<step>`; None for anything else
    (a `.tmp` being written, a `.deleting` being removed)."""
    digits = path.name[len("step_") :]
    return int(digits) if path.name.startswith("step_") and digits.isdigit() else None


class Run:
    """A run directory and its record (`record`, the parsed run.json)."""

    def __init__(self, directory: pathlib.Path, record: dict) -> None:
        self.directory, self.record = directory, record

    # ---- creating and opening

    @classmethod
    def create(
        cls,
        runs_directory: pathlib.Path,
        name: str,
        config,
        command: list[str],
        checkpoint_every: int,
        parent: dict | None = None,
        update_policies: bool = False,
    ) -> Run:
        """A new run directory with its config.json and run.json (status running, no
        segment yet). parent: the run and step a fork or a warm start began from;
        update_policies: whether every update writes its policy files (`policies/`)."""
        directory = pathlib.Path(runs_directory) / name
        directory.mkdir(parents=True)
        for part in (CHECKPOINTS, POLICIES, "league", "artifacts"):
            (directory / part).mkdir()
        (directory / "metrics.jsonl").touch()
        write_json(directory / "config.json", to_json(config))
        record = dict(
            name=name,
            status="running",
            pid=os.getpid(),
            host=socket.gethostname(),
            message=None,
            command=list(command),
            wandb_id=None,
            parent=parent,
            checkpoint_every=checkpoint_every,
            update_policies=update_policies,
            newest_checkpoint=None,
            final_checkpoint=None,
            checkpoint_ordinals={},
            probe=None,
            segments=[],
        )
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
        """Write the record to the run directory's RECORD file."""
        write_json(self.directory / RECORD, self.record)

    # ---- the record

    def start_segment(
        self, step: int, overrides: dict | None = None, rewound_from: int | None = None
    ) -> str | None:
        """Begin this process's segment at `step`, mark the run running, and return a
        warning when the commit differs from the previous segment's."""
        head, dirty = provenance.commit()
        segments = self.record["segments"]
        warning = None
        if segments and segments[-1]["commit"] != head:
            warning = (
                f"the run continues on commit {head}; its last segment ran on "
                f"{segments[-1]['commit']}"
            )
        segments.append(
            dict(
                commit=head,
                dirty=dirty,
                start_step=step,
                start_time=provenance.now(),
                end_step=None,
                end_time=None,
                overrides=dict(overrides or {}),
                rewound_from=rewound_from,
                command=list(self.record["command"]) if not segments else None,
            )
        )
        self.record.update(
            status="running", pid=os.getpid(), host=socket.gethostname(), message=None
        )
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

    def live(self) -> bool:
        """Whether the record says running and its process is alive on this host."""
        return (
            self.record["status"] == "running"
            and self.record["host"] == socket.gethostname()
            and process_alive(self.record["pid"])
        )

    def shown_status(self) -> str:
        """The status, with `running (stale)` for a running record whose process is gone
        from this host."""
        status = self.record["status"]
        if (
            status == "running"
            and self.record["host"] == socket.gethostname()
            and not process_alive(self.record["pid"])
        ):
            return "running (stale)"
        return status

    # ---- checkpoints

    @property
    def policy_path(self) -> pathlib.Path:
        """Where a BC run saves its policy file: the average its score stop read."""
        return self.directory / "policy"

    def variant_policy_path(self, variant: str) -> pathlib.Path:
        """Where a BC run saves one of the policies beside `policy/`: `policy-<variant>/`
        (`bnb.runs.session.POLICY_VARIANTS`: live, uniform, exponential)."""
        return self.directory / f"policy-{variant}"

    def update_policies_path(self, step: int) -> pathlib.Path:
        """Where the policy files written after the update that ended at environment step
        `step` live: `policies/step_<step>/{live, exponential_average, uniform_average}/`."""
        return self.directory / POLICIES / f"step_{step}"

    def checkpoint_path(self, step: int) -> pathlib.Path:
        """Where the checkpoint at environment step step lives, whether or not it exists."""
        return self.directory / CHECKPOINTS / f"step_{step}"

    def checkpoint_steps(self) -> list[int]:
        """The steps of the complete checkpoints, oldest first (a `.tmp` or a `.deleting` is
        not one)."""
        paths = (self.directory / CHECKPOINTS).iterdir()
        return sorted(step for step in map(step_of, paths) if step is not None)

    def checkpoint_at(self, step: int | None = None) -> pathlib.Path:
        """The kept checkpoint at `step`, or the newest."""
        steps = self.checkpoint_steps()
        if not steps:
            raise FileNotFoundError(f"run {self.record['name']} has no checkpoint")
        if step is None:
            return self.checkpoint_path(steps[-1])
        if step not in steps:
            raise FileNotFoundError(
                f"run {self.record['name']} keeps no checkpoint at step {step}; it keeps {steps}"
            )
        return self.checkpoint_path(step)

    def set_aside_after(self, step: int) -> list[int]:
        """For a rewind to `step`: move every newer checkpoint into
        `checkpoints/rewound_<segment>/` (and every newer update's policy files into
        `policies/rewound_<segment>/`), where they are kept and out of the retention, so the
        run can write those steps again. Returns the checkpoints' steps."""
        newer = [later for later in self.checkpoint_steps() if later > step]
        if newer:
            aside = self.directory / CHECKPOINTS / f"rewound_{len(self.record['segments'])}"
            aside.mkdir()
            for later in newer:
                self.checkpoint_path(later).rename(aside / f"step_{later}")
        policies = self.directory / POLICIES
        later_policies = (
            [path for path in policies.iterdir() if (step_of(path) or 0) > step]
            if policies.is_dir()
            else []
        )
        if later_policies:  # the rewound timeline's, out of the way of the new one's
            aside = policies / f"rewound_{len(self.record['segments'])}"
            aside.mkdir()
            for path in later_policies:
                path.rename(aside / path.name)
        if self.record["final_checkpoint"] in newer:
            self.record["final_checkpoint"] = None
        for later in newer:
            self.record["checkpoint_ordinals"].pop(str(later), None)
        return newer

    def added_checkpoint(self, step: int, final: bool = False) -> None:
        """Record the checkpoint just written at `step` (the next ordinal, unless it has one)
        and apply the retention. A deleted checkpoint is first renamed `step_<step>.deleting`,
        so a crash mid-removal leaves no directory that lists as a checkpoint; leftovers of
        an earlier crash (`.deleting`, `.tmp`) are removed here too."""
        self.record["newest_checkpoint"] = step
        if final:
            self.record["final_checkpoint"] = step
        ordinals = self.record["checkpoint_ordinals"]
        steps = self.checkpoint_steps()
        for kept in list(ordinals):
            if int(kept) not in steps:
                del ordinals[kept]
        for present in steps:  # in step order, so also one whose run.json write a crash lost
            if str(present) not in ordinals:
                ordinals[str(present)] = max(ordinals.values(), default=-1) + 1
        final_step = self.record["final_checkpoint"]
        keep = retained(ordinals.values(), ordinals.get(str(final_step)))
        for old in [present for present in steps if ordinals[str(present)] not in keep]:
            path = self.checkpoint_path(old)
            path.rename(path.with_name(path.name + ".deleting"))
            del ordinals[str(old)]
        self.save()
        for leftover in (self.directory / CHECKPOINTS).iterdir():
            if leftover.name.endswith((".deleting", ".tmp")):
                shutil.rmtree(leftover)


def list_runs(runs_directory: pathlib.Path) -> list[Run]:
    """Every run under `runs_directory`, by name."""
    directory = pathlib.Path(runs_directory)
    if not directory.is_dir():
        return []
    return [
        Run.open(directory, path.name)
        for path in sorted(directory.iterdir())
        if (path / RECORD).is_file()
    ]
