"""The small record beside a run's checkpoints that says what the run is.

Everything here is already known somewhere else. The profile, the resolved
fingerprint and the wall clock live inside the ``step_*.pt`` payload; the live
rating lives in ``elo_history.jsonl``. But answering "which of these runs do I
resume" from those means loading a 27 MB checkpoint per candidate, or parsing a
history file that grows for the whole run. This file exists so that selecting a
run costs a directory scan and one short read.

It is written where a checkpoint is written, and for the same reason: a run with
nothing resumable in it is not a run anything can select, so it does not need a
record. Status transitions update a manifest that already exists rather than
creating one.

The shape deliberately mirrors ``artifacts/store.py``'s ``artifact.json`` --
schema version, status, created/updated stamps, owner -- because a reader who
has seen one should not have to learn a second convention.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

from boost_and_broadside.artifacts.provenance import code_provenance

SCHEMA_VERSION = 1
MANIFEST_NAME = "run.json"


class RunStatus(StrEnum):
    """How the run's own process ended, as far as it was able to record."""

    RUNNING = "running"
    INTERRUPTED = "interrupted"
    COMPLETE = "complete"
    FAILED = "failed"


@lru_cache(maxsize=1)
def code_identity() -> tuple[str | None, bool | None]:
    """The commit the running code came from, and whether it was modified.

    Cached: a manifest is rewritten every update, and the checkout a live
    process runs from cannot change under it. ``None`` for both means there was
    no checkout to interrogate -- an installed wheel -- which is recorded as
    unknown rather than guessed at.
    """

    code = code_provenance()
    return code["git_commit"], code["git_dirty"]


@dataclass(frozen=True)
class RunManifest:
    """One run's selection metadata.

    Every field except ``run`` is nullable, because a trainer built without a
    resolved-configuration document or launch provenance -- tests, hermetic
    fixtures -- still writes a usable record. A reader treats a missing value as
    unknown and never as a default.
    """

    run: str
    profile: str | None = None
    status: RunStatus = RunStatus.RUNNING
    created_at: str | None = None
    updated_at: str | None = None
    global_step: int = 0
    update: int = 0
    elapsed_seconds: float | None = None
    live_elo: float | None = None
    device: str | None = None
    seed: int | None = None
    resolved_config_fingerprint: str | None = None
    wandb_run_id: str | None = None
    # Which code produced the run. A dirty checkout means the commit alone does
    # not describe what ran, so the two are only useful together.
    git_commit: str | None = None
    git_dirty: bool | None = None

    def document(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, **asdict(self), "status": str(self.status)}


def timestamp(moment: datetime | None = None) -> str:
    return (moment or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ")


def manifest_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / MANIFEST_NAME


def write_manifest(run_dir: str | Path, manifest: RunManifest) -> Path:
    """Write the manifest atomically, preserving the original ``created_at``.

    A save writes this on every update, so a torn file would be read far more
    often than it is written. Temp-then-rename means a reader sees the previous
    record or the new one, never half of either.
    """

    path = manifest_path(run_dir)
    existing = read_manifest(run_dir)
    now = timestamp()
    stamped = replace(
        manifest,
        created_at=(existing.created_at if existing is not None else None)
        or manifest.created_at
        or now,
        updated_at=now,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    payload = json.dumps(stamped.document(), indent=2, sort_keys=True) + "\n"
    with temporary.open("w") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


def read_manifest(run_dir: str | Path) -> RunManifest | None:
    """Read one run's manifest, or ``None`` when it has none or it is unreadable.

    Every run that predates this file has none, and a listing has to keep
    working for those, so an absent or damaged manifest is a normal condition
    rather than an error.
    """

    path = manifest_path(run_dir)
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(document, dict):
        return None
    fields = {field for field in RunManifest.__dataclass_fields__ if field != "run"}
    values = {key: value for key, value in document.items() if key in fields}
    status = values.pop("status", None)
    try:
        return RunManifest(
            run=str(document.get("run") or Path(run_dir).name),
            status=RunStatus(status) if status in tuple(RunStatus) else RunStatus.RUNNING,
            **values,
        )
    except TypeError:
        return None
