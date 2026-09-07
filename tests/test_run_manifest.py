"""The per-run manifest: what it records, and what it refuses to invent.

A listing reads these instead of loading checkpoints, so the failure that
matters is not a crash -- it is a manifest that reads plausibly and is wrong, or
one that exists for a run with nothing behind it.
"""

from __future__ import annotations

import json
from pathlib import Path

from boost_and_broadside.run_manifest import (
    MANIFEST_NAME,
    SCHEMA_VERSION,
    RunManifest,
    RunStatus,
    read_manifest,
    write_manifest,
)


def test_written_manifest_round_trips(tmp_path):
    manifest = RunManifest(
        run="comic-star-715",
        profile="rl",
        status=RunStatus.RUNNING,
        global_step=19_906_560,
        update=20,
        elapsed_seconds=5_821.5,
        live_elo=879.76,
        device="cuda",
        seed=1,
        wandb_run_id="abcd1234",
    )

    write_manifest(tmp_path, manifest)
    restored = read_manifest(tmp_path)

    assert restored is not None
    assert (restored.run, restored.profile, restored.status) == (
        "comic-star-715",
        "rl",
        RunStatus.RUNNING,
    )
    assert (restored.global_step, restored.update) == (19_906_560, 20)
    assert (restored.live_elo, restored.seed, restored.device) == (879.76, 1, "cuda")
    assert restored.wandb_run_id == "abcd1234"


def test_document_carries_a_schema_version_and_a_plain_status(tmp_path):
    write_manifest(tmp_path, RunManifest(run="r", status=RunStatus.COMPLETE))

    document = json.loads((tmp_path / MANIFEST_NAME).read_text())

    assert document["schema_version"] == SCHEMA_VERSION
    assert document["status"] == "complete"


def test_created_at_survives_rewrites_and_updated_at_moves(tmp_path):
    """Every save rewrites this file; the run's start time has to outlive that."""

    write_manifest(tmp_path, RunManifest(run="r", update=1))
    first = read_manifest(tmp_path)
    write_manifest(tmp_path, RunManifest(run="r", update=2))
    second = read_manifest(tmp_path)

    assert first is not None and second is not None
    assert second.created_at == first.created_at
    assert second.updated_at >= first.updated_at
    assert second.update == 2


def test_missing_and_damaged_manifests_read_as_absent(tmp_path):
    """Runs predating this file, and half-written ones, are ordinary inputs."""

    assert read_manifest(tmp_path) is None

    (tmp_path / MANIFEST_NAME).write_text("{ not json")
    assert read_manifest(tmp_path) is None

    (tmp_path / MANIFEST_NAME).write_text('["a list"]')
    assert read_manifest(tmp_path) is None


def test_unknown_and_missing_fields_do_not_break_a_reader(tmp_path):
    """A manifest written by a newer version stays readable rather than fatal."""

    (tmp_path / MANIFEST_NAME).write_text(
        json.dumps({"run": "r", "profile": "bc", "invented_later": 7})
    )

    restored = read_manifest(tmp_path)

    assert restored is not None
    assert (restored.run, restored.profile) == ("r", "bc")
    assert restored.status is RunStatus.RUNNING
    assert restored.live_elo is None


def test_a_torn_write_never_replaces_a_readable_manifest(tmp_path):
    """The temp file carries the partial content, so the real path is whole."""

    write_manifest(tmp_path, RunManifest(run="r", update=1))
    temporary = tmp_path / f".{MANIFEST_NAME}.tmp"

    assert not temporary.exists()
    assert read_manifest(tmp_path) is not None
    assert sorted(p.name for p in tmp_path.iterdir()) == [MANIFEST_NAME]


def test_run_name_falls_back_to_the_directory(tmp_path):
    (tmp_path / MANIFEST_NAME).write_text(json.dumps({"profile": "rl"}))

    restored = read_manifest(tmp_path)

    assert restored is not None
    assert restored.run == Path(tmp_path).name
