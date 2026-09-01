"""Tests for the accumulated ladder match record (live-elo-plan Phase 1)."""

import json

import numpy as np
import pytest

from boost_and_broadside.train.rl.bradley_terry import fit_bradley_terry
from boost_and_broadside.train.rl.match_matrix import SCHEMA_VERSION, MatchMatrix


class TestRecording:
    def test_a_pair_has_one_entry_however_it_is_reported(self):
        """Direction is a property of the report, not of the stored pair."""
        matrix = MatchMatrix()
        matrix.record("ckpt_2", "random", wins=7, losses=1, ties=2)
        matrix.record("random", "ckpt_2", wins=1, losses=7, ties=0)
        assert len(matrix) == 1
        assert matrix.total_games == pytest.approx(18.0)
        wins = matrix.scored_wins(["ckpt_2", "random"])
        assert wins[0, 1] == pytest.approx(14 + 1.0)  # 7+7 wins, half of 2 ties
        assert wins[1, 0] == pytest.approx(2 + 1.0)

    def test_recording_a_player_against_itself_is_rejected(self):
        matrix = MatchMatrix()
        with pytest.raises(ValueError, match="against itself"):
            matrix.record("scripted", "scripted", 1, 0, 0)

    def test_negative_counts_are_rejected(self):
        matrix = MatchMatrix()
        with pytest.raises(ValueError, match="non-negative"):
            matrix.record("a", "b", 1, -1, 0)

    def test_record_all_skips_opponents_with_no_games(self):
        matrix = MatchMatrix()
        matrix.record_all("ckpt_1", {"random": (5, 0, 0), "scripted": (0, 0, 0)})
        assert matrix.labels() == ["ckpt_1", "random"]

    def test_ties_are_games(self):
        """pair_games feeds the Fisher information, which counts every episode."""
        matrix = MatchMatrix()
        matrix.record("a", "b", wins=0, losses=0, ties=40)
        assert matrix.pair_games(["a", "b"])[0, 1] == pytest.approx(40.0)


class TestViews:
    def test_labels_absent_from_the_request_are_dropped(self):
        """A retired roster label must not silently shift the matrix indices."""
        matrix = MatchMatrix()
        matrix.record("a", "b", 5, 5, 0)
        matrix.record("a", "gone", 5, 5, 0)
        games = matrix.pair_games(["a", "b"])
        assert games.shape == (2, 2)
        assert games.sum() == pytest.approx(20.0)

    def test_restrict_drops_every_pair_touching_a_removed_player(self):
        matrix = MatchMatrix()
        matrix.record("a", "b", 5, 5, 0)
        matrix.record("b", "c", 5, 5, 0)
        assert matrix.restrict({"a", "b"}).labels() == ["a", "b"]

    def test_the_views_are_symmetric_and_hollow(self):
        matrix = MatchMatrix()
        matrix.record("a", "b", 3, 1, 2)
        matrix.record("b", "c", 4, 4, 0)
        labels = matrix.labels()
        games = matrix.pair_games(labels)
        assert np.allclose(games, games.T)
        assert np.allclose(np.diag(games), 0.0)

    def test_it_feeds_the_fitter_directly(self):
        """The point of the accumulation: a ladder refit with no glue code."""
        matrix = MatchMatrix()
        matrix.record("scripted", "random", wins=900, losses=20, ties=80)
        matrix.record("ckpt_1", "random", wins=950, losses=10, ties=40)
        matrix.record("ckpt_1", "scripted", wins=600, losses=380, ties=20)
        labels = matrix.labels()
        fit = fit_bradley_terry(matrix.scored_wins(labels), anchor=labels.index("random"))
        ratings = dict(zip(labels, fit.ratings))
        assert fit.converged
        assert ratings["ckpt_1"] > ratings["scripted"] > ratings["random"]


class TestPersistence:
    def test_it_round_trips(self, tmp_path):
        matrix = MatchMatrix()
        matrix.record("ckpt_1", "random", 900, 20, 80)
        matrix.record("ckpt_1", "scripted", 600, 380, 20)
        path = tmp_path / "match_matrix.json"
        matrix.save_json(path)
        assert MatchMatrix.load_json(path).as_records() == matrix.as_records()

    def test_a_missing_file_is_an_empty_matrix(self, tmp_path):
        """Resuming a run written before this file must not fail."""
        matrix = MatchMatrix.load_json(tmp_path / "absent.json")
        assert len(matrix) == 0
        assert matrix.total_games == pytest.approx(0.0)

    def test_a_foreign_schema_version_is_refused(self, tmp_path):
        path = tmp_path / "match_matrix.json"
        path.write_text(json.dumps({"version": SCHEMA_VERSION + 1, "pairs": []}))
        with pytest.raises(ValueError, match="schema version"):
            MatchMatrix.load_json(path)

    def test_saving_leaves_no_temporary_behind(self, tmp_path):
        matrix = MatchMatrix()
        matrix.record("a", "b", 1, 1, 0)
        matrix.save_json(tmp_path / "match_matrix.json")
        assert [p.name for p in tmp_path.iterdir()] == ["match_matrix.json"]

    def test_a_truncated_save_cannot_destroy_the_accumulation(self, tmp_path):
        """The rename is the point: a kill mid-write leaves the old file intact."""
        path = tmp_path / "match_matrix.json"
        first = MatchMatrix()
        first.record("a", "b", 10, 10, 0)
        first.save_json(path)
        path.with_suffix(".json.tmp").write_text("{ truncated")
        assert MatchMatrix.load_json(path).total_games == pytest.approx(20.0)

    def test_the_file_is_stable_across_identical_saves(self, tmp_path):
        """A churning sidecar would show up as spurious diffs in a run dir."""
        matrix = MatchMatrix()
        for player in ("c", "a", "b"):
            matrix.record(player, "random", 5, 5, 0)
        first = tmp_path / "one.json"
        second = tmp_path / "two.json"
        matrix.save_json(first)
        matrix.save_json(second)
        assert first.read_text() == second.read_text()
