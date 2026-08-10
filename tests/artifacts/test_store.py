"""Identity, atomicity, resume safety, and provenance of the artifact store."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pytest

from boost_and_broadside.artifacts import (
    STATUS_COMPLETE,
    STATUS_IN_PROGRESS,
    ArtifactError,
    ArtifactIncomplete,
    ArtifactIntegrityError,
    ArtifactRecipe,
    ArtifactRecipeMismatch,
    ArtifactStore,
    Invocation,
    artifact_digest,
    artifact_id,
    load_artifact,
    require_complete,
    verify_artifact,
)
from boost_and_broadside.artifacts.provenance import environment_provenance, normalized_command


def _recipe(**parameters) -> ArtifactRecipe:
    return ArtifactRecipe(
        artifact_type="crossover",
        result_schema_version=2,
        subjects={"run": "synthetic-run", "checkpoint_sha256": "a" * 64},
        parameters={"games": 8, **parameters},
    )


def _store(tmp_path) -> ArtifactStore:
    (tmp_path / "checkpoints" / "synthetic-run").mkdir(parents=True, exist_ok=True)
    return ArtifactStore(
        checkpoint_root=tmp_path / "checkpoints",
        standalone_root=tmp_path / "artifacts",
        invocation=Invocation(
            argv=("bnb", "crossover", "--run", "synthetic-run"),
            command="crossover",
            execution={"device": "cpu", "seed": 7},
        ),
        device="cpu",
    )


def test_recipe_digest_depends_on_every_recorded_input():
    assert _recipe().digest() == _recipe().digest()
    assert _recipe().digest() != _recipe(games=9).digest()
    assert (
        ArtifactRecipe("crossover", 2, subjects={"run": "a"}).digest()
        != ArtifactRecipe("crossover", 3, subjects={"run": "a"}).digest()
    )


def test_artifact_id_pairs_a_readable_instant_with_the_recipe_digest():
    recipe = _recipe()
    identity = artifact_id(recipe, datetime(2026, 8, 9, 14, 25, 0, tzinfo=UTC))
    assert identity == f"20260809T142500Z-{recipe.short_digest()}"


def test_artifact_type_must_be_lowercase_and_hyphenated():
    with pytest.raises(ValueError):
        ArtifactRecipe("Elo_Scale", 1)


def test_run_artifacts_live_inside_their_run_and_standalone_ones_do_not(tmp_path):
    store = _store(tmp_path)
    run_artifact = store.create(_recipe(), store.run_owner("synthetic-run"))
    standalone = store.create(_recipe(games=9), store.standalone_owner())

    assert run_artifact.path.parent.parent == (
        tmp_path / "checkpoints" / "synthetic-run" / "artifacts"
    )
    assert standalone.path.parent.parent == tmp_path / "artifacts"
    assert run_artifact.manifest["owner"] == {"kind": "run", "run": "synthetic-run"}
    assert standalone.manifest["owner"] == {"kind": "standalone", "run": None}


def test_a_missing_run_cannot_own_an_artifact(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(ArtifactError):
        store.run_owner("absent-run")


def test_owning_run_requires_exactly_one_run_behind_every_checkpoint(tmp_path):
    store = _store(tmp_path)
    checkpoints = tmp_path / "checkpoints"
    (checkpoints / "other-run").mkdir()
    first = checkpoints / "synthetic-run" / "step_1.pt"
    second = checkpoints / "other-run" / "step_1.pt"

    assert store.owning_run_for_paths([first]) == "synthetic-run"
    assert store.owning_run_for_paths([first, second]) is None
    assert store.owning_run_for_paths([tmp_path / "elsewhere.pt"]) is None
    assert store.owning_run_for_paths([]) is None


def test_manifest_records_command_execution_and_code_provenance(tmp_path):
    store = _store(tmp_path)
    artifact = store.create(_recipe(), store.standalone_owner())
    manifest = artifact.manifest

    assert manifest["invocation"]["normalized"] == "uv run bnb crossover --run synthetic-run"
    assert manifest["invocation"]["argv"][0] == "bnb"
    assert manifest["execution"] == {"device": "cpu", "seed": 7}
    assert set(manifest["code"]) == {
        "git_commit",
        "git_dirty",
        "repository_available",
        "uv_lock_sha256",
    }
    assert manifest["runtime"]["device"]["requested"] == "cpu"
    assert manifest["producer"]["name"] == "boost-and-broadside"
    assert manifest["status"] == STATUS_IN_PROGRESS
    assert manifest["recipe"]["subjects"]["run"] == "synthetic-run"


def test_environment_provenance_is_an_allowlist(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    recorded = environment_provenance()

    assert recorded["CUDA_VISIBLE_DEVICES"] == "0"
    assert not any("SECRET" in name or "KEY" in name for name in recorded)


def test_normalized_command_is_the_canonical_uv_spelling():
    assert normalized_command(["/opt/venv/bin/bnb", "elo-scale", "--run", "x"]) == (
        "uv run bnb elo-scale --run x"
    )


def test_payload_writes_record_hashes_and_sizes(tmp_path):
    store = _store(tmp_path)
    artifact = store.create(_recipe(), store.run_owner("synthetic-run"))

    artifact.write_json({"rows": [1, 2, 3]})
    artifact.write_npz({"curve": np.arange(4, dtype=np.float32)})
    artifact.write_samples_npz({"raw": np.zeros(8, dtype=np.float16)}, "errors.npz")

    recorded = {record["path"]: record for record in artifact.manifest["files"]}
    assert set(recorded) == {"result.json", "result.npz", "samples/errors.npz"}
    assert all(len(record["sha256"]) == 64 and record["bytes"] > 0 for record in recorded.values())
    assert artifact.manifest["samples_present"] is True
    verify_artifact(load_artifact(artifact.path))


def test_payload_names_cannot_escape_the_artifact_directory(tmp_path):
    store = _store(tmp_path)
    artifact = store.create(_recipe(), store.standalone_owner())
    with pytest.raises(ArtifactError):
        artifact.write_json({}, "../escaped.json")


def test_an_interrupted_write_leaves_the_previous_payload_intact(tmp_path, monkeypatch):
    store = _store(tmp_path)
    artifact = store.create(_recipe(), store.standalone_owner())
    artifact.write_json({"rows": [1]})

    def explode(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("boost_and_broadside.artifacts.store.os.replace", explode)
    with pytest.raises(OSError):
        artifact.write_json({"rows": [1, 2]})

    # The payload the reader sees is the last complete one, never a partial write.
    assert artifact.read_json() == {"rows": [1]}
    assert json.loads((artifact.path / ".result.json.tmp").read_text()) == {"rows": [1, 2]}


def test_completion_is_explicit(tmp_path):
    store = _store(tmp_path)
    artifact = store.create(_recipe(), store.standalone_owner())
    assert artifact.status == STATUS_IN_PROGRESS
    artifact.write_json({"rows": []})
    artifact.complete()
    assert load_artifact(artifact.path).status == STATUS_COMPLETE


def test_resume_continues_an_unfinished_artifact_and_never_a_finished_one(tmp_path):
    store = _store(tmp_path)
    owner = store.run_owner("synthetic-run")
    first, resumed = store.open_resumable(_recipe(), owner)
    assert resumed is False
    first.write_json({"rows": [1]})

    again, resumed = store.open_resumable(_recipe(), owner)
    assert resumed is True
    assert again.path == first.path
    assert again.read_json() == {"rows": [1]}

    again.complete()
    fresh, resumed = store.open_resumable(_recipe(), owner)
    assert resumed is False
    assert fresh.path != first.path


def test_only_a_completed_artifact_may_be_cited(tmp_path):
    store = _store(tmp_path)
    artifact = store.create(_recipe(), store.standalone_owner())
    artifact.write_json({"rows": [1]})

    # Internally consistent and hash-verifiable, but only part of what was asked.
    verify_artifact(artifact)
    with pytest.raises(ArtifactIncomplete, match="cannot be cited"):
        require_complete(artifact)

    artifact.complete()
    require_complete(load_artifact(artifact.path))


def test_resume_verifies_the_recipe_and_not_the_payload_hashes(tmp_path):
    """The recorded contract: a manifest hash may lag an interrupted payload.

    ``_write`` replaces the payload and then saves the manifest, so a process
    killed between those two steps leaves a complete, newer payload beside the
    previous recorded hash. That is exactly the artifact resume exists to
    continue, so resume checks the recipe rather than re-hashing.
    """

    store = _store(tmp_path)
    owner = store.standalone_owner()
    artifact, _ = store.open_resumable(_recipe(), owner)
    artifact.write_json({"rows": [1]})
    # The next batch lands, the process dies before the manifest is saved.
    (artifact.path / "result.json").write_text('{"rows": [1, 2]}\n')

    resumed, was_resumed = store.open_resumable(_recipe(), owner)

    assert was_resumed is True
    assert resumed.read_json() == {"rows": [1, 2]}
    # Reading the same artifact as final evidence still refuses it.
    with pytest.raises(ArtifactIntegrityError):
        load_artifact(artifact.path)


def test_a_different_recipe_never_resumes_another_measurement(tmp_path):
    store = _store(tmp_path)
    owner = store.run_owner("synthetic-run")
    first, _ = store.open_resumable(_recipe(), owner)
    second, resumed = store.open_resumable(_recipe(games=99), owner)

    assert resumed is False
    assert second.path != first.path


def test_resume_rejects_a_directory_whose_stored_recipe_was_altered(tmp_path):
    store = _store(tmp_path)
    owner = store.standalone_owner()
    artifact, _ = store.open_resumable(_recipe(), owner)

    manifest = json.loads((artifact.path / "artifact.json").read_text())
    manifest["recipe"]["parameters"]["games"] = 4096
    (artifact.path / "artifact.json").write_text(json.dumps(manifest))

    with pytest.raises(ArtifactRecipeMismatch):
        store.open_resumable(_recipe(), owner)


def test_verification_detects_an_altered_payload(tmp_path):
    store = _store(tmp_path)
    artifact = store.create(_recipe(), store.standalone_owner())
    artifact.write_json({"rows": [1]})
    (artifact.path / "result.json").write_text('{"rows": [2]}')

    with pytest.raises(ArtifactIntegrityError):
        load_artifact(artifact.path)


def test_verification_tolerates_pruned_raw_samples_but_not_missing_aggregates(tmp_path):
    store = _store(tmp_path)
    artifact = store.create(_recipe(), store.standalone_owner())
    artifact.write_json({"rows": [1]})
    artifact.write_samples_npz({"raw": np.zeros(2)}, "errors.npz")

    (artifact.path / "samples" / "errors.npz").unlink()
    verify_artifact(load_artifact(artifact.path, verify=False))

    (artifact.path / "result.json").unlink()
    with pytest.raises(ArtifactIntegrityError):
        verify_artifact(load_artifact(artifact.path, verify=False))


def test_artifact_digest_ignores_the_local_sample_payload(tmp_path):
    store = _store(tmp_path)
    artifact = store.create(_recipe(), store.standalone_owner())
    artifact.write_json({"rows": [1]})
    before = artifact_digest(artifact)
    artifact.write_samples_npz({"raw": np.zeros(2)}, "errors.npz")

    assert artifact_digest(artifact) == before


def test_an_unreadable_manifest_schema_is_refused(tmp_path):
    store = _store(tmp_path)
    artifact = store.create(_recipe(), store.standalone_owner())
    manifest = json.loads((artifact.path / "artifact.json").read_text())
    manifest["schema_version"] = 99
    (artifact.path / "artifact.json").write_text(json.dumps(manifest))

    with pytest.raises(ArtifactError):
        load_artifact(artifact.path)


def test_attaching_records_a_file_written_by_an_external_tool(tmp_path):
    store = _store(tmp_path)
    artifact = store.create(_recipe(), store.standalone_owner())
    (artifact.path / "history.jsonl").write_text('{"_step": 1}\n')

    artifact.attach("history.jsonl")

    recorded = {record["path"] for record in artifact.manifest["files"]}
    assert recorded == {"history.jsonl"}
    verify_artifact(load_artifact(artifact.path))


def test_attaching_refuses_an_absent_or_escaping_path(tmp_path):
    store = _store(tmp_path)
    artifact = store.create(_recipe(), store.standalone_owner())

    with pytest.raises(ArtifactError):
        artifact.attach("never_written.json")
    with pytest.raises(ArtifactError):
        artifact.attach("../escaped.json")
