"""What a compute mode actually leaves behind, end to end.

The store's own guarantees are unit-tested next door. This exercises a real
measurement through a real mode: that its artifact lands under the run that owns
it, carries the recipe and provenance a later reader needs, completes, and that
a second measurement of the same recipe neither overwrites nor silently reuses
the first.
"""

from __future__ import annotations

import pytest

from boost_and_broadside.artifacts import (
    STATUS_COMPLETE,
    STATUS_IN_PROGRESS,
    ArtifactStore,
    Invocation,
    load_artifact,
)
from boost_and_broadside.config.defaults import MODEL_CONFIG
from boost_and_broadside.modes.crossover import run_crossover_mode
from boost_and_broadside.smoke import build_synthetic_run


@pytest.fixture
def synthetic(tmp_path):
    return build_synthetic_run(tmp_path / "checkpoints")


def _store(tmp_path) -> ArtifactStore:
    return ArtifactStore(
        checkpoint_root=tmp_path / "checkpoints",
        standalone_root=tmp_path / "artifacts",
        invocation=Invocation(
            argv=("bnb", "crossover", "--run", "synthetic-run", "--sizes", "1"),
            command="crossover",
            execution={"device": "cpu", "seed": 7, "compile_mode": None},
        ),
        device="cpu",
    )


def _run(tmp_path, synthetic, store: ArtifactStore) -> dict:
    return run_crossover_mode(
        run_spec=synthetic.run_dir.name,
        trained_counts=[1],
        ship_config=synthetic.resolved.ship_config,
        model_config=MODEL_CONFIG,
        device="cpu",
        checkpoint_dir=str(tmp_path / "checkpoints"),
        num_envs=1,
        max_total_ships=2,
        store=store,
    )


def _artifacts(synthetic) -> list:
    root = synthetic.run_dir / "artifacts" / "crossover"
    return sorted(path for path in root.iterdir() if path.is_dir())


def test_a_measurement_is_owned_by_its_run_and_completes(tmp_path, synthetic) -> None:
    result = _run(tmp_path, synthetic, _store(tmp_path))

    written = _artifacts(synthetic)
    assert len(written) == 1
    artifact = load_artifact(written[0])
    assert artifact.status == STATUS_COMPLETE
    assert artifact.read_json() == result
    assert artifact.manifest["owner"] == {"kind": "run", "run": "synthetic-run"}


def test_the_manifest_identifies_subjects_command_and_machine(tmp_path, synthetic) -> None:
    _run(tmp_path, synthetic, _store(tmp_path))
    manifest = load_artifact(_artifacts(synthetic)[0]).manifest

    subjects = manifest["recipe"]["subjects"]
    assert subjects["run"] == "synthetic-run"
    assert len(subjects["trained"]["sha256"]) == 64
    assert subjects["trained"]["global_step"] == 1
    assert "scripted_config" in subjects["scripted"]

    training_config = subjects["trained"]["training_config"]
    assert len(training_config["resolved_config_fingerprint"]) == 64
    assert len(training_config["profile_fingerprint"]) == 64

    assert manifest["invocation"]["normalized"].startswith("uv run bnb crossover")
    assert manifest["execution"]["device"] == "cpu"
    # `wandb` is a training launch setting. A compute mode has no `--no-wandb`
    # and contacts nothing, so its artifact does not report one either way.
    assert "wandb" not in manifest["execution"]
    assert manifest["runtime"]["torch"]
    assert "git_commit" in manifest["code"]
    assert manifest["files"][0]["path"] == "result.json"


def test_repeating_a_measurement_never_overwrites_the_first(tmp_path, synthetic) -> None:
    _run(tmp_path, synthetic, _store(tmp_path))
    _run(tmp_path, synthetic, _store(tmp_path))

    written = _artifacts(synthetic)
    assert len(written) == 2
    assert all(load_artifact(path).status == STATUS_COMPLETE for path in written)


def test_an_interrupted_sweep_is_continued_rather_than_restarted(tmp_path, synthetic) -> None:
    store = _store(tmp_path)
    _run(tmp_path, synthetic, store)
    interrupted = _artifacts(synthetic)[0]

    # Reopen the finished measurement as if the process had died mid-sweep.
    manifest_path = interrupted / "artifact.json"
    manifest_path.write_text(
        manifest_path.read_text().replace(STATUS_COMPLETE, STATUS_IN_PROGRESS)
    )
    _run(tmp_path, synthetic, _store(tmp_path))

    assert _artifacts(synthetic) == [interrupted]
    assert load_artifact(interrupted).status == STATUS_COMPLETE
