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
    """One single-file, one directory, and one promoted-media renderer."""

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
    }


@pytest.fixture
def repository(tmp_path) -> Path:
    (tmp_path / "docs").mkdir()
    return tmp_path


def write_manifest(repository: Path, body: str) -> Path:
    path = repository / "docs" / "publications.toml"
    path.write_text(body)
    return path


def build_artifact(
    repository: Path,
    payload: dict,
    *,
    schema_version: int = 1,
    clean: bool = True,
    artifact_type: str = "fixture",
) -> Path:
    """A completed artifact, optionally recorded as built from a clean commit."""

    store = ArtifactStore(
        checkpoint_root=repository / "checkpoints",
        standalone_root=repository / "artifacts",
        invocation=Invocation(argv=("bnb", "fixture"), command="fixture"),
    )
    artifact = store.create(
        ArtifactRecipe(artifact_type, schema_version, subjects={"fixture": True}),
        store.standalone_owner(),
    )
    artifact.write_json(payload)
    artifact.complete()

    manifest_path = artifact.path / "artifact.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["code"] = {
        "git_commit": "0" * 40,
        "git_dirty": not clean,
        "repository_available": True,
        "uv_lock_sha256": "1" * 64,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return artifact.path.relative_to(repository)
