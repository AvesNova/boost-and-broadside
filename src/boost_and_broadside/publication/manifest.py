"""``docs/publications.toml`` — the one hand-maintained map from views to sources.

The manifest is the publication inventory. Nothing else decides what belongs in
``docs/``: not a figure count in prose, not whichever renderer happens to exist,
not what is currently on disk. An entry names a renderer, the exact artifacts it
reads, and the output it owns.

An entry whose renderer still needs a source it has not been given is
*unselected*: it is part of the inventory and its output is left alone. That is
the deliberate state of an entry whose source has not been chosen yet, and it is
reported rather than silently skipped.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from boost_and_broadside.publication import renderers as _renderers  # noqa: F401
from boost_and_broadside.publication.renderer_api import (
    PublicationError,
    Renderer,
    get_renderer,
)

MANIFEST_RELATIVE_PATH = Path("docs") / "publications.toml"
MANIFEST_SCHEMA_VERSION = 1
PUBLICATION_ROOT = Path("docs")

_ENTRY_KEYS = frozenset({"renderer", "output", "description", "artifacts", "files"})
_SHA256_LENGTH = 64


class ManifestError(PublicationError):
    """The publication manifest does not describe a renderable inventory."""


@dataclass(frozen=True)
class FileSource:
    """A local file promoted into ``docs/``, pinned by content hash."""

    path: str
    sha256: str


@dataclass(frozen=True)
class PublicationEntry:
    """One canonical output, its renderer, and the sources it is rendered from."""

    name: str
    renderer_name: str
    output: str
    description: str
    artifacts: Mapping[str, str] = field(default_factory=dict)
    files: Mapping[str, FileSource] = field(default_factory=dict)

    @property
    def renderer(self) -> Renderer:
        return get_renderer(self.renderer_name)

    @property
    def selected(self) -> bool:
        """True once every source the renderer requires has been chosen."""

        renderer = self.renderer
        return all(name in self.artifacts for name in renderer.required_artifacts) and all(
            name in self.files for name in renderer.required_files
        )

    @property
    def missing_sources(self) -> tuple[str, ...]:
        renderer = self.renderer
        return tuple(
            [name for name in renderer.required_artifacts if name not in self.artifacts]
            + [name for name in renderer.required_files if name not in self.files]
        )


@dataclass(frozen=True)
class PublicationManifest:
    """The complete inventory, in declaration order."""

    schema_version: int
    entries: tuple[PublicationEntry, ...]

    def entry(self, name: str) -> PublicationEntry:
        for entry in self.entries:
            if entry.name == name:
                return entry
        known = ", ".join(item.name for item in self.entries)
        raise ManifestError(f"unknown publication {name!r}; the manifest declares: {known}")

    @property
    def selected(self) -> tuple[PublicationEntry, ...]:
        return tuple(entry for entry in self.entries if entry.selected)


def manifest_path(root: str | Path) -> Path:
    return Path(root) / MANIFEST_RELATIVE_PATH


def load_manifest(root: str | Path) -> PublicationManifest:
    """Read and fully validate the manifest, or fail with what is wrong."""

    path = manifest_path(root)
    if not path.is_file():
        raise ManifestError(f"no publication manifest at {path}")
    try:
        document = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as error:
        raise ManifestError(f"{path} is not valid TOML: {error}") from error

    version = document.get("schema_version")
    if version != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(
            f"{path} declares manifest schema {version!r}; this build reads "
            f"{MANIFEST_SCHEMA_VERSION}"
        )
    publications = document.get("publications")
    if not isinstance(publications, dict) or not publications:
        raise ManifestError(f"{path} declares no publications")
    unexpected = sorted(set(document) - {"schema_version", "publications"})
    if unexpected:
        raise ManifestError(f"{path} has unknown top-level keys: {', '.join(unexpected)}")

    entries = tuple(
        _entry(name, body, root) for name, body in publications.items()
    )
    _reject_duplicate_outputs(entries)
    return PublicationManifest(version, entries)


def _entry(name: str, body: Any, root: str | Path) -> PublicationEntry:
    if not isinstance(body, dict):
        raise ManifestError(f"publication {name!r} is not a table")
    unexpected = sorted(set(body) - _ENTRY_KEYS)
    if unexpected:
        raise ManifestError(f"publication {name!r} has unknown keys: {', '.join(unexpected)}")
    for required in ("renderer", "output", "description"):
        if not isinstance(body.get(required), str) or not body[required]:
            raise ManifestError(f"publication {name!r} needs a non-empty {required!r}")

    renderer = get_renderer(body["renderer"])
    output = _output(name, body["output"], root)
    artifacts = _artifacts(name, body.get("artifacts", {}))
    files = _files(name, body.get("files", {}))
    _reject_unknown_sources(name, renderer, artifacts, files)
    return PublicationEntry(
        name=name,
        renderer_name=body["renderer"],
        output=output,
        description=body["description"],
        artifacts=artifacts,
        files=files,
    )


def _output(name: str, value: str, root: str | Path) -> str:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ManifestError(f"publication {name!r} output {value!r} must be a path inside docs/")
    resolved = (Path(root) / candidate).resolve()
    docs_root = (Path(root) / PUBLICATION_ROOT).resolve()
    if not resolved.is_relative_to(docs_root):
        raise ManifestError(f"publication {name!r} output {value!r} escapes {PUBLICATION_ROOT}/")
    return candidate.as_posix()


def _artifacts(name: str, value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ManifestError(f"publication {name!r} artifacts must be a table")
    for source, location in value.items():
        if not isinstance(location, str) or not location:
            raise ManifestError(
                f"publication {name!r} source {source!r} must name an artifact directory"
            )
    return dict(value)


def _files(name: str, value: Any) -> dict[str, FileSource]:
    if not isinstance(value, dict):
        raise ManifestError(f"publication {name!r} files must be a table")
    sources = {}
    for source, body in value.items():
        if not isinstance(body, dict) or set(body) != {"path", "sha256"}:
            raise ManifestError(
                f"publication {name!r} file source {source!r} needs exactly path and sha256"
            )
        digest = body["sha256"]
        if not isinstance(digest, str) or len(digest) != _SHA256_LENGTH:
            raise ManifestError(
                f"publication {name!r} file source {source!r} needs a 64-character sha256"
            )
        try:
            int(digest, 16)
        except ValueError as error:
            raise ManifestError(
                f"publication {name!r} file source {source!r} sha256 is not hexadecimal"
            ) from error
        sources[source] = FileSource(path=body["path"], sha256=digest)
    return sources


def _reject_unknown_sources(
    name: str,
    renderer: Renderer,
    artifacts: Mapping[str, str],
    files: Mapping[str, FileSource],
) -> None:
    allowed_artifacts = set(renderer.required_artifacts) | set(renderer.optional_artifacts)
    unknown = sorted(set(artifacts) - allowed_artifacts)
    if unknown:
        raise ManifestError(
            f"publication {name!r} gives renderer {renderer.name!r} unknown artifact "
            f"source(s): {', '.join(unknown)}"
        )
    unknown_files = sorted(set(files) - set(renderer.required_files))
    if unknown_files:
        raise ManifestError(
            f"publication {name!r} gives renderer {renderer.name!r} unknown file "
            f"source(s): {', '.join(unknown_files)}"
        )


def _reject_duplicate_outputs(entries: tuple[PublicationEntry, ...]) -> None:
    seen: dict[str, str] = {}
    for entry in entries:
        owner = seen.get(entry.output)
        if owner is not None:
            raise ManifestError(
                f"publications {owner!r} and {entry.name!r} both own {entry.output}"
            )
        seen[entry.output] = entry.name


__all__ = [
    "MANIFEST_RELATIVE_PATH",
    "MANIFEST_SCHEMA_VERSION",
    "FileSource",
    "ManifestError",
    "PublicationEntry",
    "PublicationManifest",
    "load_manifest",
    "manifest_path",
]
