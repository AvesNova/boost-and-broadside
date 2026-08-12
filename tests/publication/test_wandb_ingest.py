"""Ingesting an already-downloaded W&B export, offline.

The landmark run was exported before the artifact store existed, so its evidence
sits in the checkout as a plain directory. These cover the path that promotes
those bytes into a citable artifact: the payload is copied verbatim, the
provenance says where it came from, and nothing imports or contacts W&B.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from boost_and_broadside.artifacts import load_artifact, require_complete

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def ingest():
    """Load the ingestion script by path; it is a script, not a package module."""

    path = REPO_ROOT / "scripts" / "export_wandb_run.py"
    spec = importlib.util.spec_from_file_location("_export_wandb_run", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stored_export(root: Path) -> Path:
    directory = root / "wandb_export"
    (directory / "files").mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps({"profile": "rl"}))
    (directory / "summary.json").write_text(json.dumps({"overview/win_rate": 0.9}))
    (directory / "run_meta.json").write_text(
        json.dumps({"id": "abc123", "name": "fixture-run", "path": "e/p/abc123"})
    )
    (directory / "history.jsonl").write_text('{"_step": 0, "overview/kl": 0.02}\n')
    (directory / "files" / "config.yaml").write_text("seed: 7\n")
    return directory


def test_ingesting_a_stored_export_needs_no_wandb(ingest, tmp_path, monkeypatch) -> None:
    """The offline path must not import the client, let alone call it."""

    checkpoints = tmp_path / "checkpoints"
    run = checkpoints / "fixture-run"
    run.mkdir(parents=True)
    directory = _stored_export(run)
    monkeypatch.setitem(sys.modules, "wandb", None)  # importing it would raise

    path = ingest.ingest_export_directory(
        directory,
        run_name="fixture-run",
        checkpoint_root=checkpoints,
        standalone_root=tmp_path / "artifacts",
    )

    artifact = load_artifact(path)
    require_complete(artifact)
    assert artifact.artifact_type == "wandb-export"
    assert path.is_relative_to(run / "artifacts")


def test_the_ingested_payload_is_the_stored_export(ingest, tmp_path) -> None:
    checkpoints = tmp_path / "checkpoints"
    run = checkpoints / "fixture-run"
    run.mkdir(parents=True)
    directory = _stored_export(run)

    path = ingest.ingest_export_directory(
        directory,
        run_name="fixture-run",
        checkpoint_root=checkpoints,
        standalone_root=tmp_path / "artifacts",
    )

    artifact = load_artifact(path)
    recorded = {record["path"] for record in artifact.manifest["files"]}
    assert recorded == {
        "config.json",
        "summary.json",
        "run_meta.json",
        "history.jsonl",
        "files/config.yaml",
    }
    # history.jsonl is what every training figure reads; it is copied, not reformatted.
    assert (path / "history.jsonl").read_bytes() == (directory / "history.jsonl").read_bytes()


def test_the_recipe_records_where_the_export_came_from(ingest, tmp_path) -> None:
    """An ingested export is not a fresh measurement and does not claim to be."""

    checkpoints = tmp_path / "checkpoints"
    (checkpoints / "fixture-run").mkdir(parents=True)
    directory = _stored_export(checkpoints / "fixture-run")

    path = ingest.ingest_export_directory(
        directory,
        run_name="fixture-run",
        checkpoint_root=checkpoints,
        standalone_root=tmp_path / "artifacts",
    )

    recipe = load_artifact(path).manifest["recipe"]
    assert recipe["subjects"] == {"wandb_run": "e/p/abc123", "run": "fixture-run"}
    # The sampling W&B was asked for is not in the files; unknown, not assumed.
    assert recipe["parameters"] == {"samples": None}
    assert recipe["sources"] == {"export_directory": directory.as_posix()}


def test_a_directory_that_is_not_an_export_is_refused(ingest, tmp_path) -> None:
    incomplete = tmp_path / "wandb_export"
    incomplete.mkdir()
    (incomplete / "config.json").write_text("{}")

    with pytest.raises(FileNotFoundError, match="summary.json"):
        ingest.ingest_export_directory(
            incomplete, checkpoint_root=tmp_path / "checkpoints",
            standalone_root=tmp_path / "artifacts",
        )
