"""Exact run and distinct checkpoint-selection policy contracts."""

import os

import pytest

from boost_and_broadside.evaluation.run_catalog import (
    CheckpointKind,
    CheckpointNotFoundError,
    RunNotFoundError,
    resolve_exact_run,
    resolve_explicit_checkpoint,
    resolve_legacy_capture_run,
    resolve_legacy_elo_run,
    select_final_training_checkpoint,
    select_latest_resumable_checkpoint,
    select_named_best_policy,
    select_tournament_ladder_policies,
)


def _run(tmp_path, name="run-a"):
    path = tmp_path / name
    path.mkdir()
    return path


def test_exact_run_resolution_never_interprets_sentinels_or_paths(tmp_path):
    latest = _run(tmp_path, "latest")
    assert resolve_exact_run("latest", tmp_path).path == latest
    with pytest.raises(RunNotFoundError):
        resolve_exact_run("missing", tmp_path)
    with pytest.raises(RunNotFoundError):
        resolve_exact_run("nested/run", tmp_path)


@pytest.mark.parametrize("resolver", [resolve_legacy_capture_run, resolve_legacy_elo_run])
def test_transitional_latest_adapters_raise_typed_errors_for_missing_roots(tmp_path, resolver):
    with pytest.raises(RunNotFoundError):
        resolver("latest", tmp_path / "missing")


def test_numeric_step_policy_ignores_mtime_and_lexicographic_width(tmp_path):
    run = _run(tmp_path)
    high = run / "step_100.pt"
    low = run / "step_000009.pt"
    unrelated = run / "step_latest.pt"
    for checkpoint in (high, low, unrelated):
        checkpoint.touch()
    os.utime(low, (500, 500))
    os.utime(high, (100, 100))

    resumable = select_latest_resumable_checkpoint(run)
    final = select_final_training_checkpoint(run)
    assert (resumable.path, resumable.step, resumable.kind) == (
        high,
        100,
        CheckpointKind.RESUMABLE,
    )
    assert (final.path, final.step, final.kind) == (high, 100, CheckpointKind.FINAL)


def test_named_best_and_explicit_checkpoint_policies_do_not_fall_back(tmp_path):
    run = _run(tmp_path)
    best = run / "best_training.pt"
    explicit = run / "custom.pt"
    best.touch()
    explicit.touch()
    assert select_named_best_policy(run, "training").path == best
    assert resolve_explicit_checkpoint(explicit).kind is CheckpointKind.EXPLICIT
    with pytest.raises(CheckpointNotFoundError):
        select_named_best_policy(run, "avg")


def test_ladder_selection_uses_roster_metadata_and_numeric_order(tmp_path):
    run = _run(tmp_path)
    late = run / "ladder_step_100.pt"
    early = run / "ladder_step_20.pt"
    late.touch()
    early.touch()
    roster = {
        "entries": [
            {"kind": "scripted", "label": "scripted"},
            {
                "kind": "checkpoint",
                "label": "late",
                "path": str(late),
                "global_step": 100,
                "elo": 1200.0,
            },
            {
                "kind": "checkpoint",
                "label": "missing",
                "path": str(run / "gone.pt"),
                "global_step": 50,
                "elo": 1100.0,
            },
            {
                "kind": "checkpoint",
                "label": "early",
                "path": str(early),
                "global_step": 20,
                "elo": 1000.0,
            },
        ]
    }
    selected = select_tournament_ladder_policies(run, roster)
    assert [(item.label, item.global_step) for item in selected] == [("early", 20), ("late", 100)]
    assert all(item.checkpoint.kind is CheckpointKind.LADDER for item in selected)
