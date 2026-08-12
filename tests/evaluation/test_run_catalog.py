"""Exact run and distinct checkpoint-selection policy contracts."""

import os

import pytest

from boost_and_broadside.evaluation.run_catalog import (
    CheckpointKind,
    CheckpointNotFoundError,
    InvalidCheckpointError,
    RunNotFoundError,
    resolve_exact_run,
    resolve_explicit_checkpoint,
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
    _run(tmp_path, "latest")
    with pytest.raises(RunNotFoundError):
        resolve_exact_run("latest", tmp_path)
    with pytest.raises(RunNotFoundError):
        resolve_exact_run("missing", tmp_path)
    with pytest.raises(RunNotFoundError):
        resolve_exact_run("nested/run", tmp_path)


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
                "label": "ckpt_100",
                "path": str(late),
                "global_step": 100,
                "elo": 1200.0,
            },
            {
                "kind": "checkpoint",
                "label": "ckpt_50",
                "path": str(run / "ladder_step_000000000050.pt"),
                "global_step": 50,
                "elo": 1100.0,
            },
            {
                "kind": "checkpoint",
                "label": "ckpt_20",
                "path": str(early),
                "global_step": 20,
                "elo": 1000.0,
            },
        ]
    }
    selected = select_tournament_ladder_policies(run, roster)
    assert [(item.label, item.global_step) for item in selected] == [
        ("ckpt_20", 20),
        ("ckpt_100", 100),
    ]
    assert all(item.checkpoint.kind is CheckpointKind.LADDER for item in selected)


def test_ladder_selection_never_accepts_an_existing_foreign_roster_path(tmp_path):
    run = _run(tmp_path)
    foreign_dir = _run(tmp_path, "other")
    foreign = foreign_dir / "ladder_step_000000000020.pt"
    foreign.touch()
    roster = {
        "entries": [
            {
                "kind": "checkpoint",
                "label": "ckpt_20",
                "path": str(foreign),
                "global_step": 20,
                "elo": 1000.0,
            }
        ]
    }

    assert select_tournament_ladder_policies(run, roster) == []

    local = run / foreign.name
    local.touch()
    selected = select_tournament_ladder_policies(run, roster)
    assert [item.checkpoint.path for item in selected] == [local]


@pytest.mark.parametrize(
    ("label", "path", "step"),
    [
        ("ckpt_21", "ladder_step_000000000020.pt", 20),
        ("ckpt_20", "ladder_step_000000000021.pt", 20),
        ("ckpt_20", "foreign.pt", 20),
        ("ckpt_20", None, 20),
    ],
)
def test_ladder_selection_rejects_mismatched_recorded_identity(tmp_path, label, path, step):
    run = _run(tmp_path)
    if path is not None:
        (run / path).touch()
    roster = {
        "entries": [
            {
                "kind": "checkpoint",
                "label": label,
                "path": path,
                "global_step": step,
                "elo": 1000.0,
            }
        ]
    }

    with pytest.raises(InvalidCheckpointError):
        select_tournament_ladder_policies(run, roster)


def _recorded_run(tmp_path, name, *, profile=None, step=None, modified=None):
    """A run directory as a listing would find it on disk."""
    from boost_and_broadside.run_manifest import RunManifest, write_manifest

    path = tmp_path / name
    path.mkdir()
    if step is not None:
        (path / f"step_{step:012d}.pt").touch()
    if profile is not None:
        write_manifest(path, RunManifest(run=name, profile=profile, update=3, live_elo=880.0))
    if modified is not None:
        os.utime(path, (modified, modified))
    return path


def test_runs_are_summarized_newest_first(tmp_path):
    from boost_and_broadside.evaluation.run_catalog import summarize_runs

    _recorded_run(tmp_path, "oldest", profile="rl", step=1, modified=1_000)
    _recorded_run(tmp_path, "middle", profile="rl", step=2, modified=2_000)
    _recorded_run(tmp_path, "newest", profile="bc", step=3, modified=3_000)

    summaries = summarize_runs(tmp_path)

    assert [summary.run.name for summary in summaries] == ["newest", "middle", "oldest"]
    assert [summary.profile for summary in summaries] == ["bc", "rl", "rl"]
    assert [summary.latest_step for summary in summaries] == [3, 2, 1]


def test_a_limit_stops_the_scan_rather_than_trimming_it(tmp_path):
    """Hundreds of run directories must not each cost a manifest read."""
    from boost_and_broadside.evaluation.run_catalog import summarize_runs

    for index in range(25):
        _recorded_run(tmp_path, f"run-{index:02d}", profile="rl", step=index + 1, modified=index)

    summaries = summarize_runs(tmp_path, limit=10)

    assert len(summaries) == 10
    assert summaries[0].run.name == "run-24"


def test_unknown_profile_runs_never_satisfy_a_profile_filter(tmp_path):
    """A run written before the manifest existed says nothing about its profile,
    and guessing is how a resume lands in the wrong launch."""
    from boost_and_broadside.evaluation.run_catalog import summarize_runs

    _recorded_run(tmp_path, "unrecorded", step=1, modified=3_000)
    _recorded_run(tmp_path, "recorded", profile="rl", step=2, modified=1_000)

    assert [s.run.name for s in summarize_runs(tmp_path, profile="rl")] == ["recorded"]
    assert [s.run.name for s in summarize_runs(tmp_path)] == ["unrecorded", "recorded"]


def test_resumable_filter_skips_runs_with_no_step_checkpoint(tmp_path):
    from boost_and_broadside.evaluation.run_catalog import summarize_runs

    _recorded_run(tmp_path, "ladder-only", profile="rl", modified=3_000)
    _recorded_run(tmp_path, "resumable", profile="rl", step=7, modified=1_000)

    summaries = summarize_runs(tmp_path, resumable_only=True)

    assert [summary.run.name for summary in summaries] == ["resumable"]
    assert summaries[0].resumable


def test_latest_resumable_run_picks_the_newest_of_its_own_profile(tmp_path):
    from boost_and_broadside.evaluation.run_catalog import select_latest_resumable_run

    _recorded_run(tmp_path, "newest-bc", profile="bc", step=9, modified=4_000)
    _recorded_run(tmp_path, "newest-rl-unresumable", profile="rl", modified=3_000)
    _recorded_run(tmp_path, "older-rl", profile="rl", step=5, modified=2_000)

    assert select_latest_resumable_run("rl", tmp_path).name == "older-rl"
    assert select_latest_resumable_run("bc", tmp_path).name == "newest-bc"


def test_latest_resumable_run_explains_itself_when_nothing_matches(tmp_path):
    from boost_and_broadside.evaluation.run_catalog import select_latest_resumable_run

    _recorded_run(tmp_path, "unrecorded", step=1)

    with pytest.raises(RunNotFoundError, match="bnb runs"):
        select_latest_resumable_run("rl", tmp_path)


def test_summarizing_a_missing_checkpoint_root_is_empty_not_an_error(tmp_path):
    from boost_and_broadside.evaluation.run_catalog import summarize_runs

    assert summarize_runs(tmp_path / "absent") == []
