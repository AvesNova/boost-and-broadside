"""Unit tests for the Elo league roster: sampling, ladder, policy cache, persistence."""

import json

import pytest
import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.train.rl.roster import EloRoster

# The cache tests pre-set entry.policy, so no file is ever read; these only have
# to satisfy the signature.
_LOAD_ARGS = dict(ship_config=ShipConfig(), num_ships=4, device="cpu", team_pma_k=())


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
        assert roster.sample(live_elo=1000.0) is None

    def test_random_anchor_is_never_sampled(self):
        torch.manual_seed(0)
        roster = _make_roster()
        roster.add_checkpoint(path="/ckpt/a.pt", global_step=1, update=1, initial_elo=1000.0)
        sampled = {roster.sample(live_elo=1000.0).label for _ in range(50)}
        assert sampled == {"ckpt_1"}

    def test_frozen_checkpoints_remain_sampleable(self):
        torch.manual_seed(0)
        roster = _make_roster(uniform_sampling=True)
        _add_frozen_checkpoint(roster, step=1, elo=1000.0)
        roster.add_checkpoint(path="/ckpt/b.pt", global_step=2, update=2, initial_elo=1000.0)
        sampled = {roster.sample(live_elo=1000.0).label for _ in range(200)}
        assert sampled == {"ckpt_1", "ckpt_2"}

    def test_proximity_sampling_prefers_near_elo_entries(self):
        """With a huge Elo gap the far entry's weight underflows to ~0, so the
        near-Elo entry is sampled every time."""
        torch.manual_seed(0)
        roster = _make_roster()
        _add_frozen_checkpoint(roster, step=1, elo=1000.0)
        roster.add_checkpoint(path="/ckpt/far.pt", global_step=2, update=2, initial_elo=51000.0)
        sampled = {roster.sample(live_elo=1000.0).label for _ in range(100)}
        assert sampled == {"ckpt_1"}

    def test_uniform_sampling_ignores_elo_proximity(self):
        torch.manual_seed(0)
        roster = _make_roster(uniform_sampling=True)
        _add_frozen_checkpoint(roster, step=1, elo=1000.0)
        roster.add_checkpoint(path="/ckpt/far.pt", global_step=2, update=2, initial_elo=51000.0)
        sampled = {roster.sample(live_elo=1000.0).label for _ in range(200)}
        assert sampled == {"ckpt_1", "ckpt_2"}

    def test_scripted_and_avg_are_ordinary_candidates(self):
        """Every opponent type is drawn from one pool; nothing is special-cased."""
        torch.manual_seed(0)
        roster = _make_roster(uniform_sampling=True)
        roster.add_special("scripted", initial_elo=1000.0)
        roster.add_special("avg", initial_elo=1000.0)
        _add_frozen_checkpoint(roster, step=1, elo=1000.0)
        sampled = {roster.sample(live_elo=1000.0).label for _ in range(300)}
        assert sampled == {"scripted", "avg", "ckpt_1"}

    def test_scripted_fades_as_the_live_rating_outruns_it(self):
        """The opponent curriculum is the rating, not a schedule.

        At equal rating the scripted agent is a live candidate; far above it, it
        stops being drawn in favour of the nearer entry.
        """
        torch.manual_seed(0)
        roster = _make_roster()
        roster.add_special("scripted", initial_elo=1000.0)
        roster.add_special("avg", initial_elo=2500.0)
        assert "scripted" in {roster.sample(live_elo=1000.0).label for _ in range(50)}
        assert {roster.sample(live_elo=2500.0).label for _ in range(50)} == {"avg"}

    def test_retired_entries_are_not_sampled_but_keep_their_rating(self):
        torch.manual_seed(0)
        roster = _make_roster(uniform_sampling=True)
        keep = _add_frozen_checkpoint(roster, step=1, elo=1000.0)
        drop = roster.add_checkpoint(path="/ckpt/b.pt", global_step=2, update=2, initial_elo=1200.0)
        roster.retire(drop)
        sampled = {roster.sample(live_elo=1000.0).label for _ in range(100)}
        assert sampled == {keep.label}
        assert drop in roster.entries and drop.elo == 1200.0

    def test_sample_returns_none_when_every_entry_is_retired(self):
        roster = _make_roster()
        entry = roster.add_checkpoint(
            path="/ckpt/a.pt", global_step=1, update=1, initial_elo=1000.0
        )
        roster.retire(entry)
        assert roster.sample(live_elo=1000.0) is None


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
        # Stationary references first, then the newest `count` checkpoints.
        assert [e.label for e in anchors] == ["random", "ckpt_2", "ckpt_3"]

    def test_stationary_references_never_age_out_of_the_ladder(self):
        """Checkpoints rotate; fixed players are permanent calibration points.

        A rung's rating is a measured property of a stationary agent, so
        dropping it would discard a reference the run cannot regenerate.
        """
        roster = _make_roster()
        roster.add_special("scripted", initial_elo=1000.0)
        roster.add_reference(p_scripted=0.5, elo=200.0)
        for step in range(1, 6):
            _add_frozen_checkpoint(roster, step=step, elo=100.0 * step)

        anchors = roster.ladder_anchors(2)
        labels = [e.label for e in anchors]
        assert labels[:3] == ["random", "semi_scripted_0p5", "scripted"]  # sorted by elo
        assert labels[3:] == ["ckpt_4", "ckpt_5"]
        assert all(e.is_stationary for e in anchors[:3])

    def test_ladder_anchors_with_no_checkpoints_is_stationary_only(self):
        roster = _make_roster()
        roster.add_reference(p_scripted=0.5, elo=200.0)
        assert [e.label for e in roster.ladder_anchors(2)] == ["random", "semi_scripted_0p5"]

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

    def test_set_special_elo_tracks_the_evaluator(self):
        """Proximity sampling reads these ratings, so a stale one misdirects it."""
        roster = _make_roster()
        entry = roster.add_special("avg", initial_elo=500.0)
        roster.set_special_elo("avg", 1450.0)
        assert entry.elo == 1450.0

    def test_set_special_elo_is_a_noop_before_the_entry_exists(self):
        roster = _make_roster()
        roster.set_special_elo("avg", 1450.0)  # avg only joins once it accumulates
        assert all(e.kind != "avg" for e in roster.entries)


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


class TestSemiRandomProbabilityRoundTrip:
    """A rung that loses p_scripted plays as random while keeping its rating.

    This cost run 728 its whole ladder. save_json never wrote the field and
    load_json never read it, so every resume rebuilt the rungs with
    p_scripted=None -- which the evaluator plays as the uniform random agent.
    The rungs kept ratings of 200-950 while being trivially beatable, so the
    live rating was pulled up by sweeps that looked like beating a 950. Nothing
    in any metric showed it: the run's own win rates against the rungs were the
    only evidence, and they read as the policy having improved.
    """

    @staticmethod
    def roster_with_rungs() -> EloRoster:
        roster = EloRoster()
        for probability in (0.2, 0.5, 0.95):
            roster.add_reference(probability, 1000.0 * probability)
        return roster

    def test_the_probability_survives_a_save_and_load(self, tmp_path):
        original = self.roster_with_rungs()
        path = tmp_path / "roster.json"
        original.save_json(path)
        restored = EloRoster()
        restored.load_json(path)
        before = {e.label: e.p_scripted for e in original.entries if e.kind == "semi_random"}
        after = {e.label: e.p_scripted for e in restored.entries if e.kind == "semi_random"}
        assert after == before
        assert all(value is not None for value in after.values())

    def test_a_roster_written_before_the_field_existed_is_repaired(self, tmp_path):
        """Every run on disk predates the fix, so the label has to be enough."""
        path = tmp_path / "roster.json"
        path.write_text(
            json.dumps(
                {
                    "entries": [
                        {
                            "kind": "semi_random",
                            "label": "semi_scripted_0p95",
                            "elo": 950.0,
                            "global_step": 0,
                            "update": 0,
                            "path": None,
                            "fixed": True,
                        }
                    ]
                }
            )
        )
        roster = EloRoster()
        roster.load_json(path)
        assert roster.entries[0].p_scripted == pytest.approx(0.95)

    def test_non_rung_entries_keep_no_probability(self, tmp_path):
        roster = EloRoster()
        path = tmp_path / "roster.json"
        roster.save_json(path)
        restored = EloRoster()
        restored.load_json(path)
        assert all(
            entry.p_scripted is None
            for entry in restored.entries
            if entry.kind != "semi_random"
        )
