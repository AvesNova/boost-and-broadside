"""The artifact store: durable, self-describing evidence with managed paths.

Every compute mode writes here instead of choosing its own output file. An
artifact directory holds the publishable aggregates, its complete manifest, and
optionally a raw sample payload that is never tracked:

    checkpoints/<run>/artifacts/<type>/<id>/
        artifact.json    recipe, provenance, schemas, and per-file hashes
        result.json      publishable aggregates and report inputs
        result.npz       bounded numeric arrays where JSON is inappropriate
        samples/         optional compressed raw samples; always gitignored

Ownership follows the subjects: a measurement about one exact run belongs to
that run, and one without a single owning run lands under the standalone
``artifacts/`` root. Nothing here writes under ``docs/`` — publication is the
only writer of canonical views.

Writes are atomic and the manifest is rewritten after every payload file, so an
interrupted measurement leaves either the previous consistent state or the new
one. Resuming re-verifies the stored recipe field by field before continuing,
which is what stops a resumed sweep from mixing two different requests.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from boost_and_broadside.artifacts.identity import (
    ARTIFACT_MANIFEST_SCHEMA_VERSION,
    ArtifactRecipe,
    artifact_id,
)
from boost_and_broadside.artifacts.provenance import (
    code_provenance,
    environment_provenance,
    normalized_command,
    producer_provenance,
    runtime_provenance,
)
from boost_and_broadside.errors import UserFacingError

_MANIFEST_NAME = "artifact.json"
_SAMPLES_DIR = "samples"
_RUN_ARTIFACT_DIR = "artifacts"

STATUS_IN_PROGRESS = "in-progress"
STATUS_COMPLETE = "complete"


class ArtifactError(UserFacingError):
    """Base error for the artifact store."""


class ArtifactRecipeMismatch(ArtifactError):
    """A stored artifact does not describe the measurement asking to resume it."""


class ArtifactIntegrityError(ArtifactError):
    """A stored artifact's payload does not match its recorded hashes."""


@dataclass(frozen=True)
class ArtifactOwner:
    """Where an artifact lives, and which run (if any) owns it."""

    kind: str
    root: Path
    run: str | None = None

    def type_root(self, artifact_type: str) -> Path:
        return self.root / artifact_type


@dataclass(frozen=True)
class Invocation:
    """The command and validated execution settings behind one measurement."""

    argv: tuple[str, ...] = ()
    command: str | None = None
    execution: Mapping[str, Any] = field(default_factory=dict)

    def document(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "argv": list(self.argv),
            "normalized": normalized_command(self.argv) if self.argv else None,
        }


def _now() -> datetime:
    return datetime.now(UTC)


def _timestamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Replace ``path`` with ``payload`` or leave the previous content intact."""

    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def file_sha256(path: str | Path) -> str:
    """Content hash of a file, streamed so a large checkpoint stays cheap."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_sha256 = file_sha256


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=False) + "\n").encode("utf-8")


class Artifact:
    """One artifact directory, its manifest, and the writes that update it."""

    def __init__(self, path: Path, manifest: dict[str, Any]) -> None:
        self.path = path
        self._manifest = manifest

    # -- identity ---------------------------------------------------------

    @property
    def artifact_id(self) -> str:
        return self._manifest["artifact_id"]

    @property
    def artifact_type(self) -> str:
        return self._manifest["artifact_type"]

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._manifest))

    @property
    def status(self) -> str:
        return self._manifest["status"]

    @property
    def samples_dir(self) -> Path:
        return self.path / _SAMPLES_DIR

    # -- reads ------------------------------------------------------------

    def has(self, name: str) -> bool:
        return any(record["path"] == name for record in self._manifest["files"])

    def read_json(self, name: str = "result.json") -> dict[str, Any]:
        return json.loads((self.path / name).read_text())

    # -- writes -----------------------------------------------------------

    def write_json(self, payload: Any, name: str = "result.json") -> Path:
        """Write a JSON payload and record its hash in the manifest."""

        return self._write(name, _json_bytes(payload))

    def write_jsonl(self, rows: Sequence[Mapping[str, Any]], name: str) -> Path:
        body = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
        return self._write(name, body)

    def write_npz(self, arrays: Mapping[str, np.ndarray], name: str = "result.npz") -> Path:
        """Write bounded numeric arrays that do not belong in JSON."""

        return self._write(name, _compressed_npz(arrays))

    def write_samples_npz(self, arrays: Mapping[str, np.ndarray], name: str) -> Path:
        """Write an optional raw sample payload; ``samples/`` is never tracked."""

        self.samples_dir.mkdir(parents=True, exist_ok=True)
        return self._write(f"{_SAMPLES_DIR}/{name}", _compressed_npz(arrays))

    def attach(self, name: str) -> Path:
        """Record a file an external tool wrote into this artifact directory.

        Ingestion from another system (a W&B export downloading a run's files)
        cannot route every byte through :meth:`_write`. Attaching hashes what
        landed, so the manifest still describes the complete payload.
        """

        path = self.path / name
        if not path.resolve().is_relative_to(self.path.resolve()):
            raise ArtifactError(f"artifact payload {name!r} escapes {self.path}")
        if not path.is_file():
            raise ArtifactError(f"no file {name!r} in {self.path} to attach")
        self._record_file(name, path)
        self._save_manifest()
        return path

    def complete(self) -> Path:
        """Mark the measurement finished; only then is the artifact citable."""

        self._manifest["status"] = STATUS_COMPLETE
        return self._save_manifest()

    # -- internals --------------------------------------------------------

    def _write(self, name: str, payload: bytes) -> Path:
        path = self.path / name
        if not path.resolve().is_relative_to(self.path.resolve()):
            raise ArtifactError(f"artifact payload {name!r} escapes {self.path}")
        _atomic_write_bytes(path, payload)
        self._record_file(name, path)
        self._save_manifest()
        return path

    def _record_file(self, name: str, path: Path) -> None:
        record = {"path": name, "sha256": _sha256(path), "bytes": path.stat().st_size}
        files = [entry for entry in self._manifest["files"] if entry["path"] != name]
        files.append(record)
        self._manifest["files"] = sorted(files, key=lambda entry: entry["path"])
        self._manifest["samples_present"] = any(
            entry["path"].startswith(f"{_SAMPLES_DIR}/") for entry in self._manifest["files"]
        )

    def _save_manifest(self) -> Path:
        self._manifest["updated_at"] = _timestamp(_now())
        path = self.path / _MANIFEST_NAME
        _atomic_write_bytes(path, _json_bytes(self._manifest))
        return path


def _compressed_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
    import io

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{key: np.asarray(value) for key, value in arrays.items()})
    return buffer.getvalue()


class ArtifactStore:
    """Creates and resumes artifacts under the managed roots for one invocation."""

    def __init__(
        self,
        *,
        checkpoint_root: str | Path = "checkpoints",
        standalone_root: str | Path = "artifacts",
        invocation: Invocation | None = None,
        device: str | None = None,
    ) -> None:
        self.checkpoint_root = Path(checkpoint_root)
        self.standalone_root = Path(standalone_root)
        self.invocation = invocation or Invocation()
        self.device = device

    # -- ownership --------------------------------------------------------

    def run_owner(self, run: str) -> ArtifactOwner:
        """Durable evidence about one exact run belongs inside that run."""

        run_dir = self.checkpoint_root / run
        if not run_dir.is_dir():
            raise ArtifactError(f"no run directory {run_dir} to own this artifact")
        return ArtifactOwner("run", run_dir / _RUN_ARTIFACT_DIR, run)

    def standalone_owner(self) -> ArtifactOwner:
        """Evidence without one owning run lives under the standalone root."""

        return ArtifactOwner("standalone", self.standalone_root, None)

    def owner_for(self, run: str | None) -> ArtifactOwner:
        return self.standalone_owner() if run is None else self.run_owner(run)

    def owning_run_for_paths(self, paths: Sequence[str | Path]) -> str | None:
        """The single run that owns every given checkpoint, or None.

        A measurement over two runs' checkpoints is a property of neither, and a
        measurement against scripted opponents may have no checkpoint at all.
        Both land in the standalone root rather than being filed under a run
        that only partly explains them.
        """

        root = self.checkpoint_root.resolve()
        runs = set()
        for candidate in paths:
            path = Path(candidate).resolve()
            if not path.is_relative_to(root):
                return None
            relative = path.relative_to(root)
            if len(relative.parts) < 2:
                return None
            runs.add(relative.parts[0])
        if len(runs) != 1:
            return None
        return runs.pop()

    # -- creation and resume ---------------------------------------------

    def create(self, recipe: ArtifactRecipe, owner: ArtifactOwner) -> Artifact:
        """Start a new artifact; an existing measurement is never overwritten.

        Identities carry one-second resolution, so a repeat of the same recipe
        inside the same second is recorded at the next free instant. That keeps
        directory names unique and sortable, and keeps ``created_at`` equal to
        the instant the identity names.
        """

        created = _now()
        type_root = owner.type_root(recipe.artifact_type)
        while (path := type_root / artifact_id(recipe, created)).exists():
            created = created.replace(microsecond=0) + timedelta(seconds=1)
        path.mkdir(parents=True)
        artifact = Artifact(path, self._new_manifest(recipe, owner, created))
        artifact._save_manifest()
        return artifact

    def open_resumable(
        self, recipe: ArtifactRecipe, owner: ArtifactOwner
    ) -> tuple[Artifact, bool]:
        """Continue the newest unfinished artifact for this exact recipe.

        A finished artifact is never reopened: a repeated measurement is a new
        measurement, with its own seeds and provenance. Only an interrupted one
        is continued, and only after its stored recipe matches this one exactly.
        """

        existing = self._latest_matching(recipe, owner)
        if existing is None:
            return self.create(recipe, owner), False
        return existing, True

    def _latest_matching(self, recipe: ArtifactRecipe, owner: ArtifactOwner) -> Artifact | None:
        type_root = owner.type_root(recipe.artifact_type)
        if not type_root.is_dir():
            return None
        suffix = f"-{recipe.short_digest()}"
        for candidate in sorted(type_root.iterdir(), reverse=True):
            if not candidate.is_dir() or not candidate.name.endswith(suffix):
                continue
            artifact = load_artifact(candidate, verify=False)
            if artifact.manifest["recipe"] != recipe.canonical():
                raise ArtifactRecipeMismatch(
                    f"{candidate} shares this recipe digest but describes a different "
                    "measurement; refusing to resume it"
                )
            if artifact.status == STATUS_IN_PROGRESS:
                return artifact
        return None

    def _new_manifest(
        self, recipe: ArtifactRecipe, owner: ArtifactOwner, created: datetime
    ) -> dict[str, Any]:
        return {
            "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
            "artifact_type": recipe.artifact_type,
            "artifact_id": artifact_id(recipe, created),
            "status": STATUS_IN_PROGRESS,
            "created_at": _timestamp(created),
            "updated_at": _timestamp(created),
            "owner": {"kind": owner.kind, "run": owner.run},
            "recipe": recipe.canonical(),
            "recipe_digest": recipe.digest(),
            "result_schema_version": recipe.result_schema_version,
            "invocation": self.invocation.document(),
            "execution": dict(self.invocation.execution),
            "code": code_provenance(),
            "runtime": runtime_provenance(self.device),
            "environment": environment_provenance(),
            "producer": producer_provenance(),
            "files": [],
            "samples_present": False,
        }


def load_artifact(path: str | Path, *, verify: bool = True) -> Artifact:
    """Read an artifact from disk, optionally re-hashing every recorded file.

    Publication verifies; the store itself does not, because a resumed
    measurement is about to rewrite the payload it would have checked.
    """

    directory = Path(path)
    manifest_path = directory / _MANIFEST_NAME
    if not manifest_path.is_file():
        raise ArtifactError(f"no {_MANIFEST_NAME} in {directory}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != ARTIFACT_MANIFEST_SCHEMA_VERSION:
        raise ArtifactError(
            f"{directory} uses artifact manifest schema "
            f"{manifest.get('schema_version')!r}; this build reads "
            f"{ARTIFACT_MANIFEST_SCHEMA_VERSION}"
        )
    artifact = Artifact(directory, manifest)
    if verify:
        verify_artifact(artifact)
    return artifact


def verify_artifact(artifact: Artifact) -> None:
    """Re-hash every recorded payload file and reject a partial or altered one."""

    for record in artifact.manifest["files"]:
        path = artifact.path / record["path"]
        if not path.is_file():
            if record["path"].startswith(f"{_SAMPLES_DIR}/"):
                # Raw samples are deliberately local and may have been pruned;
                # the aggregates the manifest also records remain sufficient.
                continue
            raise ArtifactIntegrityError(f"{artifact.path} is missing {record['path']}")
        if _sha256(path) != record["sha256"]:
            raise ArtifactIntegrityError(
                f"{artifact.path}/{record['path']} does not match its recorded hash"
            )


def artifact_digest(artifact: Artifact) -> str:
    """A stable digest over an artifact's identity and its recorded file hashes."""

    payload = {
        "artifact_id": artifact.artifact_id,
        "recipe_digest": artifact.manifest["recipe_digest"],
        "files": [
            {"path": record["path"], "sha256": record["sha256"]}
            for record in artifact.manifest["files"]
            if not record["path"].startswith(f"{_SAMPLES_DIR}/")
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def discard_artifact(artifact: Artifact) -> None:
    """Remove an artifact directory that could not be completed."""

    shutil.rmtree(artifact.path, ignore_errors=True)


__all__ = [
    "Artifact",
    "ArtifactError",
    "ArtifactIntegrityError",
    "ArtifactOwner",
    "ArtifactRecipeMismatch",
    "ArtifactStore",
    "Invocation",
    "STATUS_COMPLETE",
    "STATUS_IN_PROGRESS",
    "artifact_digest",
    "discard_artifact",
    "file_sha256",
    "load_artifact",
    "verify_artifact",
]
