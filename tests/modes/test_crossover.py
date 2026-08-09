"""Tests for crossover result persistence."""

import pytest

from boost_and_broadside.artifacts import ArtifactRecipe, ArtifactStore
from boost_and_broadside.modes.crossover import _curve_rate, _load_progress, _outcome_record


def test_outcome_record_preserves_all_results() -> None:
    record = _outcome_record(wins=7, losses=2, ties=1, games=10, mean_episode_steps=123.5)

    assert record == {
        "counts_available": True,
        "wins": 7,
        "losses": 2,
        "ties": 1,
        "games": 10,
        "win_rate": 0.7,
        "mean_episode_steps": 123.5,
    }


def test_outcome_record_rejects_inconsistent_game_count() -> None:
    with pytest.raises(ValueError, match="do not match"):
        _outcome_record(wins=7, losses=2, ties=1, games=11, mean_episode_steps=123.5)


def test_load_progress_marks_legacy_rates_as_missing_counts(tmp_path) -> None:
    store = ArtifactStore(checkpoint_root=tmp_path, standalone_root=tmp_path / "artifacts")
    artifact = store.create(ArtifactRecipe("crossover", 2), store.standalone_owner())
    artifact.write_json({"rows": [{"trained": 4, "curve": {"5": 0.75}}]})

    row = _load_progress(artifact)[4]

    assert row["curve"]["5"] == {"counts_available": False, "win_rate": 0.75}
    assert _curve_rate(row["curve"]["5"]) == 0.75
