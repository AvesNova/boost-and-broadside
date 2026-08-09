"""Typed exact-run discovery and checkpoint-selection policies."""

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

_STEP_PATTERN = re.compile(r"step_(?P<step>\d+)\.pt", re.ASCII)


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
    training_elo: float


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
    """Select existing roster entries eligible for a stationary tournament."""
    run_path = run.path if isinstance(run, RunRef) else run
    selected: list[LadderPolicyRef] = []
    for entry in sorted(
        (item for item in roster.get("entries", []) if item.get("kind") == "checkpoint"),
        key=lambda item: int(item["global_step"]),
    ):
        recorded = Path(entry["path"])
        path = recorded if recorded.is_file() else run_path / recorded.name
        if not path.is_file():
            continue
        step = int(entry["global_step"])
        selected.append(
            LadderPolicyRef(
                checkpoint=CheckpointRef(path, CheckpointKind.LADDER, step=step),
                label=str(entry["label"]),
                global_step=step,
                training_elo=float(entry["elo"]),
            )
        )
    return selected
