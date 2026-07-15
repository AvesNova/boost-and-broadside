"""Unit tests for the ELO league roster: sampling, eviction, persistence."""

import pytest
import torch

from boost_and_broadside.train.rl.roster import EloRoster


def _make_roster(max_size: int = 3, **overrides) -> EloRoster:
    defaults = dict(max_size=max_size, k_factor=32.0, elo_temperature=200.0)
    defaults.update(overrides)
    return EloRoster(**defaults)


class TestSampling:
    def test_sample_returns_none_when_all_entries_fixed(self):
        roster = _make_roster()
        for e in roster.entries:
            e.fixed = True
        assert roster.sample(training_elo=1000.0) is None

    def test_fixed_entries_are_never_sampled(self):
        torch.manual_seed(0)
        roster = _make_roster()
        roster.entries[0].fixed = True  # exclude the random anchor
        roster.add_checkpoint(path="/ckpt/a.pt", global_step=1, update=1, initial_elo=1000.0)
        sampled = {roster.sample(training_elo=1000.0).label for _ in range(50)}
        assert sampled == {"ckpt_1"}

    def test_proximity_sampling_prefers_near_elo_entries(self):
        """With a huge ELO gap the far entry's weight underflows to ~0, so the
        near-ELO entry is sampled every time."""
        torch.manual_seed(0)
        roster = _make_roster()
        roster.entries[0].fixed = True  # exclude the random anchor
        roster.add_checkpoint(path="/ckpt/near.pt", global_step=1, update=1, initial_elo=1000.0)
        roster.add_checkpoint(path="/ckpt/far.pt", global_step=2, update=2, initial_elo=51000.0)
        sampled = {roster.sample(training_elo=1000.0).label for _ in range(100)}
        assert sampled == {"ckpt_1"}

    def test_uniform_sampling_ignores_elo_proximity(self):
        torch.manual_seed(0)
        roster = _make_roster(uniform_sampling=True)
        roster.entries[0].fixed = True  # exclude the random anchor
        roster.add_checkpoint(path="/ckpt/near.pt", global_step=1, update=1, initial_elo=1000.0)
        roster.add_checkpoint(path="/ckpt/far.pt", global_step=2, update=2, initial_elo=51000.0)
        sampled = {roster.sample(training_elo=1000.0).label for _ in range(200)}
        assert sampled == {"ckpt_1", "ckpt_2"}


class TestEviction:
    def test_lowest_elo_checkpoint_evicted_over_capacity(self):
        roster = _make_roster(max_size=2)
        roster.add_checkpoint(path="/ckpt/a.pt", global_step=1, update=1, initial_elo=100.0)
        roster.add_checkpoint(path="/ckpt/b.pt", global_step=2, update=2, initial_elo=50.0)
        roster.add_checkpoint(path="/ckpt/c.pt", global_step=3, update=3, initial_elo=200.0)
        ckpt_labels = {e.label for e in roster.entries if e.kind == "checkpoint"}
        assert ckpt_labels == {"ckpt_1", "ckpt_3"}

    def test_special_entries_do_not_count_toward_capacity(self):
        roster = _make_roster(max_size=1)
        roster.add_special("avg", initial_elo=500.0)
        roster.add_special("scripted", initial_elo=500.0)
        roster.add_checkpoint(path="/ckpt/a.pt", global_step=1, update=1, initial_elo=100.0)
        kinds = sorted(e.kind for e in roster.entries)
        assert kinds == ["avg", "checkpoint", "random", "scripted"]

    def test_add_special_is_idempotent(self):
        roster = _make_roster()
        first = roster.add_special("avg", initial_elo=500.0)
        second = roster.add_special("avg", initial_elo=999.0)
        assert second is first
        assert sum(e.kind == "avg" for e in roster.entries) == 1

    def test_add_special_rejects_unknown_kind(self):
        roster = _make_roster()
        with pytest.raises(AssertionError):
            roster.add_special("checkpoint")


class TestPersistence:
    def test_save_load_round_trip(self, tmp_path):
        roster = _make_roster()
        roster.add_special("avg", global_step=10, update=2, initial_elo=750.0)
        roster.add_checkpoint(path="/ckpt/a.pt", global_step=20, update=4, initial_elo=800.0)
        path = tmp_path / "roster.json"
        roster.save_json(path)

        restored = _make_roster()
        restored.load_json(path)
        original = [
            (e.kind, e.label, e.elo, e.global_step, e.update, e.path, e.fixed)
            for e in roster.entries
        ]
        loaded = [
            (e.kind, e.label, e.elo, e.global_step, e.update, e.path, e.fixed)
            for e in restored.entries
        ]
        assert loaded == original

    def test_load_json_replaces_existing_entries(self, tmp_path):
        roster = _make_roster()
        path = tmp_path / "roster.json"
        roster.save_json(path)  # only the random anchor

        restored = _make_roster()
        restored.add_checkpoint(path="/ckpt/stale.pt", global_step=1, update=1)
        restored.load_json(path)
        assert [e.kind for e in restored.entries] == ["random"]

    def test_kept_paths_returns_checkpoint_paths_only(self):
        roster = _make_roster()
        roster.add_special("avg")
        roster.add_checkpoint(path="/ckpt/a.pt", global_step=1, update=1)
        roster.add_checkpoint(path="/ckpt/b.pt", global_step=2, update=2)
        assert roster.kept_paths() == {"/ckpt/a.pt", "/ckpt/b.pt"}
