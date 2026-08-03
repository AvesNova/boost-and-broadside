"""Unit tests for the Elo league roster: sampling, ladder, policy cache, persistence."""

import pytest
import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.train.rl.roster import EloRoster

# The cache tests pre-set entry.policy, so no file is ever read; these only have
# to satisfy the signature.
_LOAD_ARGS = dict(ship_config=ShipConfig(), num_ships=4, device="cpu")


def _make_roster(max_size: int = 3, **overrides) -> EloRoster:
    defaults = dict(max_size=max_size, elo_temperature=200.0)
    defaults.update(overrides)
    return EloRoster(**defaults)


def _add_frozen_checkpoint(roster: EloRoster, step: int, elo: float):
    """Add a checkpoint and immediately freeze it (ladder fast-forward)."""
    entry = roster.add_checkpoint(
        path=f"/ckpt/{step}.pt", global_step=step, update=step, initial_elo=elo
    )
    roster.freeze_floating()
    return entry


class TestSampling:
    def test_sample_returns_none_with_only_random_anchor(self):
        roster = _make_roster()
        assert roster.sample(training_elo=1000.0) is None

    def test_random_anchor_is_never_sampled(self):
        torch.manual_seed(0)
        roster = _make_roster()
        roster.add_checkpoint(path="/ckpt/a.pt", global_step=1, update=1, initial_elo=1000.0)
        sampled = {roster.sample(training_elo=1000.0).label for _ in range(50)}
        assert sampled == {"ckpt_1"}

    def test_frozen_checkpoints_remain_sampleable(self):
        torch.manual_seed(0)
        roster = _make_roster(uniform_sampling=True)
        _add_frozen_checkpoint(roster, step=1, elo=1000.0)
        roster.add_checkpoint(path="/ckpt/b.pt", global_step=2, update=2, initial_elo=1000.0)
        sampled = {roster.sample(training_elo=1000.0).label for _ in range(200)}
        assert sampled == {"ckpt_1", "ckpt_2"}

    def test_proximity_sampling_prefers_near_elo_entries(self):
        """With a huge Elo gap the far entry's weight underflows to ~0, so the
        near-Elo entry is sampled every time."""
        torch.manual_seed(0)
        roster = _make_roster()
        _add_frozen_checkpoint(roster, step=1, elo=1000.0)
        roster.add_checkpoint(path="/ckpt/far.pt", global_step=2, update=2, initial_elo=51000.0)
        sampled = {roster.sample(training_elo=1000.0).label for _ in range(100)}
        assert sampled == {"ckpt_1"}

    def test_uniform_sampling_ignores_elo_proximity(self):
        torch.manual_seed(0)
        roster = _make_roster(uniform_sampling=True)
        _add_frozen_checkpoint(roster, step=1, elo=1000.0)
        roster.add_checkpoint(path="/ckpt/far.pt", global_step=2, update=2, initial_elo=51000.0)
        sampled = {roster.sample(training_elo=1000.0).label for _ in range(200)}
        assert sampled == {"ckpt_1", "ckpt_2"}


class TestLadder:
    def test_random_anchor_starts_frozen_at_zero(self):
        roster = _make_roster()
        random_entry = roster.entries[0]
        assert random_entry.kind == "random"
        assert random_entry.fixed
        assert random_entry.elo == 0.0

    def test_new_checkpoint_is_floating(self):
        roster = _make_roster()
        entry = roster.add_checkpoint(path="/ckpt/a.pt", global_step=1, update=1, initial_elo=200.0)
        assert roster.floating_checkpoint() is entry

    def test_add_checkpoint_rejects_second_floating(self):
        roster = _make_roster()
        roster.add_checkpoint(path="/ckpt/a.pt", global_step=1, update=1)
        with pytest.raises(AssertionError, match="freeze"):
            roster.add_checkpoint(path="/ckpt/b.pt", global_step=2, update=2)

    def test_freeze_floating_fixes_current_rating(self):
        roster = _make_roster()
        entry = roster.add_checkpoint(path="/ckpt/a.pt", global_step=1, update=1, initial_elo=200.0)
        roster.set_floating_elo(187.5)
        frozen = roster.freeze_floating()
        assert frozen is entry
        assert frozen.fixed
        assert frozen.elo == 187.5
        assert roster.floating_checkpoint() is None

    def test_ladder_anchors_is_random_only_before_any_freeze(self):
        roster = _make_roster()
        anchors = roster.ladder_anchors(2)
        assert [e.kind for e in anchors] == ["random"]

    def test_ladder_anchors_returns_newest_frozen_tail_oldest_first(self):
        roster = _make_roster()
        _add_frozen_checkpoint(roster, step=1, elo=200.0)
        _add_frozen_checkpoint(roster, step=2, elo=400.0)
        _add_frozen_checkpoint(roster, step=3, elo=600.0)
        anchors = roster.ladder_anchors(2)
        assert [e.label for e in anchors] == ["ckpt_2", "ckpt_3"]

    def test_entries_are_never_evicted(self):
        roster = _make_roster(max_size=2)
        for step in range(1, 5):
            _add_frozen_checkpoint(roster, step=step, elo=100.0 * step)
        ckpt_labels = {e.label for e in roster.entries if e.kind == "checkpoint"}
        assert ckpt_labels == {"ckpt_1", "ckpt_2", "ckpt_3", "ckpt_4"}


class TestPolicyCache:
    def test_least_recently_used_policy_unloaded_beyond_max_size(self):
        roster = _make_roster(max_size=2)
        entries = [_add_frozen_checkpoint(roster, step=s, elo=100.0 * s) for s in range(1, 4)]
        for entry in entries:
            entry.policy = object()  # pretend the weights are already loaded
            roster.load_policy(entry, **_LOAD_ARGS)
        assert entries[0].policy is None
        assert entries[1].policy is not None
        assert entries[2].policy is not None

    def test_touching_a_loaded_policy_refreshes_its_cache_position(self):
        roster = _make_roster(max_size=2)
        entries = [_add_frozen_checkpoint(roster, step=s, elo=100.0 * s) for s in range(1, 4)]
        for entry in entries[:2]:
            entry.policy = object()
            roster.load_policy(entry, **_LOAD_ARGS)
        roster.load_policy(entries[0], **_LOAD_ARGS)  # refresh the oldest
        entries[2].policy = object()
        roster.load_policy(entries[2], **_LOAD_ARGS)
        assert entries[0].policy is not None
        assert entries[1].policy is None

    def test_evict_all_checkpoint_policies_clears_cache(self):
        roster = _make_roster()
        entry = _add_frozen_checkpoint(roster, step=1, elo=100.0)
        entry.policy = object()
        roster.load_policy(entry, **_LOAD_ARGS)
        roster.evict_all_checkpoint_policies()
        assert entry.policy is None
        assert roster._load_order == []


class TestSpecialEntries:
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
        _add_frozen_checkpoint(roster, step=20, elo=800.0)
        roster.add_checkpoint(path="/ckpt/float.pt", global_step=30, update=6, initial_elo=900.0)
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
        _add_frozen_checkpoint(roster, step=1, elo=0.0)
        roster.add_checkpoint(path="/ckpt/b.pt", global_step=2, update=2)
        assert roster.kept_paths() == {"/ckpt/1.pt", "/ckpt/b.pt"}
