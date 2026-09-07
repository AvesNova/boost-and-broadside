"""What a renderer is, and the only way ``bnb figures`` talks to one.

A renderer turns stored artifacts into files. It never decides *where* those
files go, never reads the repository, and never opens a socket: it is handed
already-verified sources and a directory, and it writes into that directory.

Declaring sources by name and schema is what lets ``bnb figures`` check a run
before rendering it, so a missing measurement or one written under a shape the
renderer does not read is named up front rather than crashing mid-render.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from boost_and_broadside.artifacts import Artifact
from boost_and_broadside.errors import UserFacingError


class PublicationError(UserFacingError):
    """A rendering failure the CLI reports as one concise line."""


@dataclass(frozen=True)
class RenderInputs:
    """Verified sources, keyed by the names the figure set used."""

    artifacts: Mapping[str, Artifact] = field(default_factory=dict)

    def artifact(self, name: str) -> Artifact:
        try:
            return self.artifacts[name]
        except KeyError as error:  # pragma: no cover - the figure set is checked first
            raise PublicationError(f"renderer asked for absent source {name!r}") from error

    def optional(self, name: str) -> Artifact | None:
        return self.artifacts.get(name)


@dataclass(frozen=True)
class Renderer:
    """One named, versioned way to turn artifacts into canonical files."""

    name: str
    description: str
    render: Callable[[RenderInputs, Path], list[Path]]
    required_artifacts: tuple[str, ...] = ()
    optional_artifacts: tuple[str, ...] = ()
    # Source name → the ``result_schema_version`` values this renderer reads.
    # A source outside its list is refused before rendering rather than
    # producing a figure from a shape the renderer does not actually understand.
    supported_schemas: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    # A directory renderer owns a whole tree and prunes what it no longer
    # produces; a single-file renderer must produce exactly one file.
    multi_file: bool = False

    @property
    def source_names(self) -> tuple[str, ...]:
        return (*self.required_artifacts, *self.optional_artifacts)


_REGISTRY: dict[str, Renderer] = {}


def register(renderer: Renderer) -> Renderer:
    """Add a renderer to the registry; names are unique and versioned."""

    if renderer.name in _REGISTRY:
        raise PublicationError(f"renderer {renderer.name!r} is already registered")
    _REGISTRY[renderer.name] = renderer
    return renderer


def get_renderer(name: str) -> Renderer:
    try:
        return _REGISTRY[name]
    except KeyError as error:
        known = ", ".join(sorted(_REGISTRY)) or "none"
        raise PublicationError(f"unknown renderer {name!r}; registered: {known}") from error


def registered_renderers() -> Mapping[str, Renderer]:
    return dict(_REGISTRY)


__all__ = [
    "PublicationError",
    "RenderInputs",
    "Renderer",
    "get_renderer",
    "register",
    "registered_renderers",
]
