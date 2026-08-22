"""Fixture renderers and artifacts for the publication engine's own tests.

These deliberately do not use the real renderers: what is under test here is the
manifest contract and the install/check machinery, and a fixture renderer can be
made to misbehave in ways a real one never should.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from boost_and_broadside.artifacts import ArtifactRecipe, ArtifactStore, Invocation
from boost_and_broadside.publication import renderer_api
from boost_and_broadside.publication.renderer_api import Renderer, register


@pytest.fixture
def isolated_registry():
    """Give one test its own renderer registry.

    Not autouse: the inventory tests read the shipped manifest and need the real
    renderers registered.
    """

    saved = dict(renderer_api._REGISTRY)
    renderer_api._REGISTRY.clear()
    try:
        yield renderer_api._REGISTRY
    finally:
        renderer_api._REGISTRY.clear()
        renderer_api._REGISTRY.update(saved)


def _render_summary(inputs, out_dir: Path) -> list[Path]:
    payload = inputs.artifact("measurement").read_json()
    path = out_dir / "summary.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return [path]


def _render_report(inputs, out_dir: Path) -> list[Path]:
    payload = inputs.artifact("measurement").read_json()
    written = []
    for name in payload["sections"]:
        path = out_dir / f"{name}.md"
        path.write_text(f"# {name}\n")
        written.append(path)
    return written


def _render_promoted(inputs, out_dir: Path) -> list[Path]:
    path = out_dir / "clip.gif"
    path.write_bytes(inputs.files["clip"].read_bytes())
    return [path]


@pytest.fixture
def renderers(isolated_registry):
    """One single-file, one directory, one promoted-media, one figure-copy."""

    return {
        "summary": register(
            Renderer(
                name="fixture-summary-v1",
                description="Copy a measurement's aggregates.",
                render=_render_summary,
                required_artifacts=("measurement",),
                supported_schemas={"measurement": (1,)},
            )
        ),
        "report": register(
            Renderer(
                name="fixture-report-v1",
                description="Write a small report tree.",
                render=_render_report,
                required_artifacts=("measurement",),
                supported_schemas={"measurement": (1,)},
                multi_file=True,
            )
        ),
        "promoted": register(
            Renderer(
                name="fixture-media-v1",
                description="Promote a local clip.",
                render=_render_promoted,
                required_files=("clip",),
            )
        ),
        "figure": register(
            Renderer(
                name="fixture-figure-copy-v1",
                description="Copy one entry of a run's rendered figure set.",
                render=_render_summary,
                required_artifacts=("figures",),
                names_a_figure=True,
            )
        ),
    }


@pytest.fixture
def repository(tmp_path) -> Path:
    (tmp_path / "docs").mkdir()
    return tmp_path


def write_manifest(repository: Path, body: str) -> Path:
    path = repository / "docs" / "publications.toml"
    path.write_text(body)
    return path


def fixture_store(repository: Path) -> ArtifactStore:
    return ArtifactStore(
        checkpoint_root=repository / "checkpoints",
        standalone_root=repository / "artifacts",
        invocation=Invocation(argv=("bnb", "fixture"), command="fixture"),
    )


def record_commit(artifact_path: Path, *, clean: bool = True) -> None:
    """Rewrite an artifact's code provenance; the test tree is not a repository.

    Call after the last payload write: the artifact holds its manifest in memory
    and would write the real provenance back over this.
    """

    manifest_path = artifact_path / "artifact.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["code"] = {
        "git_commit": "0" * 40,
        "git_dirty": not clean,
        "repository_available": True,
        "uv_lock_sha256": "1" * 64,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))


def build_artifact(
    repository: Path,
    payload: dict,
    *,
    schema_version: int = 1,
    clean: bool = True,
    artifact_type: str = "fixture",
    complete: bool = True,
) -> Path:
    """A completed artifact, optionally recorded as built from a clean commit."""

    store = fixture_store(repository)
    artifact = store.create(
        ArtifactRecipe(artifact_type, schema_version, subjects={"fixture": True}),
        store.standalone_owner(),
    )
    artifact.write_json(payload)
    if complete:
        artifact.complete()
    record_commit(artifact.path, clean=clean)
    return artifact.path.relative_to(repository)
