"""Typed exact-run discovery and checkpoint-selection policies."""

import os
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from boost_and_broadside.run_manifest import RunManifest, read_manifest

_STEP_PATTERN = re.compile(r"step_(?P<step>\d+)\.pt", re.ASCII)
_LADDER_STEP_PATTERN = re.compile(r"ladder_step_(?P<step>\d+)\.pt", re.ASCII)


class RunCatalogError(Exception):
    """Base class for run/checkpoint catalog failures."""


class RunNotFoundError(FileNotFoundError, RunCatalogError):
    """An exact named run does not exist."""


class CheckpointNotFoundError(FileNotFoundError, RunCatalogError):
    """A requested checkpoint policy has no matching file."""


class InvalidCheckpointError(ValueError, RunCatalogError):
    """A checkpoint path/name does not satisfy its selection policy."""


class CheckpointKind(StrEnum):
    RESUMABLE = "resumable"
    FINAL = "final"
    BEST = "best"
    LADDER = "ladder"
    EXPLICIT = "explicit"


@dataclass(frozen=True)
class RunRef:
    """An exact run name and its resolved directory."""

    name: str
    path: Path


@dataclass(frozen=True)
class CheckpointRef:
    """A checkpoint selected under one explicit policy."""

    path: Path
    kind: CheckpointKind
    step: int | None = None
    name: str | None = None


@dataclass(frozen=True)
class LadderPolicyRef:
    """A roster-declared tournament checkpoint and its rating metadata."""

    checkpoint: CheckpointRef
    label: str
    global_step: int
    live_elo: float


def resolve_exact_run(run_name: str, checkpoint_dir: str | Path = "checkpoints") -> RunRef:
    """Resolve one exact run name below ``checkpoint_dir``."""
    if (
        not run_name
        or Path(run_name).name != run_name
        or run_name in {".", "..", "latest", "none"}
    ):
        raise RunNotFoundError(f"invalid exact run name: {run_name!r}")
    path = Path(checkpoint_dir) / run_name
    if not path.is_dir():
        raise RunNotFoundError(f"run directory not found: {path}")
    return RunRef(run_name, path)


def _numeric_steps(run: RunRef | Path) -> list[tuple[int, Path]]:
    path = run.path if isinstance(run, RunRef) else run
    candidates: list[tuple[int, Path]] = []
    for checkpoint in path.glob("step_*.pt"):
        match = _STEP_PATTERN.fullmatch(checkpoint.name)
        if match is not None:
            candidates.append((int(match.group("step")), checkpoint))
    return candidates


def select_latest_resumable_checkpoint(run: RunRef | Path) -> CheckpointRef:
    """Select the greatest numeric ``step_*.pt`` within one exact run."""
    path = run.path if isinstance(run, RunRef) else run
    candidates = _numeric_steps(run)
    if not candidates:
        raise CheckpointNotFoundError(f"no resumable step_*.pt checkpoint in {path}")
    step, checkpoint = max(candidates, key=lambda item: item[0])
    return CheckpointRef(checkpoint, CheckpointKind.RESUMABLE, step=step)


@dataclass(frozen=True)
class RunSummary:
    """One run as a listing sees it, without opening a checkpoint."""

    run: RunRef
    manifest: RunManifest | None
    latest_step: int | None
    modified: float

    @property
    def profile(self) -> str | None:
        return self.manifest.profile if self.manifest is not None else None

    @property
    def resumable(self) -> bool:
        return self.latest_step is not None


def _run_directories(checkpoint_dir: str | Path) -> list[tuple[float, Path]]:
    """Every run directory with its modification time, newest first.

    Deliberately a bare scan: no file inside any run is opened here. With
    hundreds of run directories, ordering has to be settled before anything is
    read, so that reading is confined to the few a caller actually takes.
    """

    root = Path(checkpoint_dir)
    if not root.is_dir():
        return []
    found: list[tuple[float, Path]] = []
    with os.scandir(root) as entries:
        for entry in entries:
            if entry.is_dir():
                found.append((entry.stat().st_mtime, Path(entry.path)))
    found.sort(key=lambda item: (-item[0], item[1].name))
    return found


def summarize_runs(
    checkpoint_dir: str | Path = "checkpoints",
    *,
    limit: int | None = None,
    profile: str | None = None,
    resumable_only: bool = False,
) -> list[RunSummary]:
    """Summarize the most recently touched runs, newest first.

    Reads lazily in modification order and stops once ``limit`` matches are in
    hand, so a filtered listing costs the runs it had to look at rather than the
    whole directory. A run whose profile is unknown -- anything written before
    the manifest existed -- never satisfies a profile filter, because guessing
    which profile an old run belongs to is exactly the mistake that would resume
    the wrong thing.
    """

    summaries: list[RunSummary] = []
    for modified, path in _run_directories(checkpoint_dir):
        manifest = read_manifest(path)
        if profile is not None and (manifest is None or manifest.profile != profile):
            continue
        steps = _numeric_steps(path)
        latest_step = max((step for step, _ in steps), default=None)
        if resumable_only and latest_step is None:
            continue
        summaries.append(RunSummary(RunRef(path.name, path), manifest, latest_step, modified))
        if limit is not None and len(summaries) >= limit:
            break
    return summaries


def select_latest_resumable_run(
    profile: str, checkpoint_dir: str | Path = "checkpoints"
) -> RunRef:
    """The most recently touched run of ``profile`` that has something to resume.

    Scoped to the profile because resuming an RL run into a BC launch, or the
    reverse, is a mistake the resume itself would only catch as configuration
    drift.
    """

    found = summarize_runs(checkpoint_dir, limit=1, profile=profile, resumable_only=True)
    if not found:
        raise RunNotFoundError(
            f"no resumable {profile} run found under {Path(checkpoint_dir)}; "
            "runs recorded before run.json existed have no profile and are not "
            "eligible -- name one explicitly with --resume, and see bnb runs"
        )
    return found[0].run


def select_final_training_checkpoint(run: RunRef | Path) -> CheckpointRef:
    """Select the numerically final training step under the final-policy contract."""
    selected = select_latest_resumable_checkpoint(run)
    return CheckpointRef(selected.path, CheckpointKind.FINAL, step=selected.step)


def select_named_best_policy(run: RunRef | Path, name: str) -> CheckpointRef:
    """Select ``best_<name>.pt`` without falling back to another policy family."""
    if not name or Path(name).name != name or not re.fullmatch(r"[a-z0-9_]+", name):
        raise InvalidCheckpointError(f"invalid best-policy name: {name!r}")
    path = (run.path if isinstance(run, RunRef) else run) / f"best_{name}.pt"
    if not path.is_file():
        raise CheckpointNotFoundError(f"named best policy not found: {path}")
    return CheckpointRef(path, CheckpointKind.BEST, name=name)


def resolve_explicit_checkpoint(path: str | Path) -> CheckpointRef:
    """Validate an explicit checkpoint file path."""
    checkpoint = Path(path)
    if checkpoint.suffix != ".pt" or not checkpoint.is_file():
        raise CheckpointNotFoundError(f"checkpoint not found: {checkpoint}")
    match = _STEP_PATTERN.fullmatch(checkpoint.name)
    step = int(match.group("step")) if match is not None else None
    return CheckpointRef(checkpoint, CheckpointKind.EXPLICIT, step=step)


def select_tournament_ladder_policies(
    run: RunRef | Path, roster: dict
) -> list[LadderPolicyRef]:
    """Select roster checkpoints strictly from one exact run.

    Roster paths are historical metadata and may be absolute paths from another
    machine. Their basename identifies the checkpoint, but only the matching file
    below ``run`` is eligible. The production ladder filename, label, and recorded
    global step must agree before a policy can enter a tournament.
    """
    run_path = (run.path if isinstance(run, RunRef) else run).resolve()

    def recorded_step(entry: dict) -> int:
        try:
            return int(entry["global_step"])
        except (KeyError, TypeError, ValueError) as error:
            raise InvalidCheckpointError(
                f"roster checkpoint has invalid global_step: {entry.get('global_step')!r}"
            ) from error

    entries = roster.get("entries")
    if not isinstance(entries, list) or any(not isinstance(item, dict) for item in entries):
        raise InvalidCheckpointError("roster entries must be a list of objects")
    selected: list[LadderPolicyRef] = []
    for entry in sorted(
        (item for item in entries if item.get("kind") == "checkpoint"),
        key=recorded_step,
    ):
        step = recorded_step(entry)
        path_text = entry.get("path")
        if not isinstance(path_text, str) or not path_text:
            raise InvalidCheckpointError(
                f"roster checkpoint step {step} has invalid path {path_text!r}"
            )
        recorded = Path(path_text)
        match = _LADDER_STEP_PATTERN.fullmatch(recorded.name)
        if match is None or int(match.group("step")) != step:
            raise InvalidCheckpointError(
                f"roster checkpoint {recorded!s} does not identify recorded step {step}"
            )
        expected_label = f"ckpt_{step}"
        if entry.get("label") != expected_label:
            raise InvalidCheckpointError(
                f"roster checkpoint step {step} has label {entry.get('label')!r}; "
                f"expected {expected_label!r}"
            )
        path = run_path / recorded.name
        if not path.is_file():
            continue
        if path.resolve().parent != run_path:
            raise InvalidCheckpointError(
                f"roster checkpoint resolves outside selected run {run_path}: {path}"
            )
        try:
            live_elo = float(entry["elo"])
        except (KeyError, TypeError, ValueError) as error:
            raise InvalidCheckpointError(
                f"roster checkpoint step {step} has invalid Elo {entry.get('elo')!r}"
            ) from error
        selected.append(
            LadderPolicyRef(
                checkpoint=CheckpointRef(path, CheckpointKind.LADDER, step=step),
                label=str(entry["label"]),
                global_step=step,
                live_elo=live_elo,
            )
        )
    return selected
