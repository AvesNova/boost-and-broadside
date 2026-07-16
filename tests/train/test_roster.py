"""Tests for the v2 league roster, PFSP curriculum, and retention policy."""

import json

import pytest
import torch

from boost_and_broadside.train.rl.roster import LeagueRoster


def _make_roster(league_size: int = 8, **overrides) -> LeagueRoster:
    defaults = {
        "league_size": league_size,
        "pfsp_mode": "hard",
        "pfsp_exponent": 2.0,
        "admission_prior_games": 10.0,
    }
    defaults.update(overrides)
    return LeagueRoster(**defaults)


def test_random_and_scripted_are_first_class_sampleable_members() -> None:
    roster = _make_roster()

    sampled = roster.sample_opponents(4, roster.ratings)

    assert {entry.kind for entry in sampled} == {"random", "scripted"}


def test_hard_pfsp_prefers_opponent_live_rarely_beats() -> None:
    torch.manual_seed(0)
    roster = _make_roster()
    roster.refresh_ratings({"live": 1_000.0, "random": -2_000.0, "scripted": 2_000.0})

    sampled = [roster.sample_opponents(1)[0].kind for _ in range(50)]

    assert set(sampled) == {"scripted"}


def test_variance_pfsp_prefers_near_even_opponent() -> None:
    torch.manual_seed(0)
    roster = _make_roster(pfsp_mode="variance")
    roster.refresh_ratings({"live": 0.0, "random": -2_000.0, "scripted": 0.0})

    sampled = [roster.sample_opponents(1)[0].kind for _ in range(50)]

    assert set(sampled) == {"scripted"}


def test_pfsp_samples_without_replacement() -> None:
    roster = _make_roster()
    roster.add_avg(global_step=10, update=2)

    sampled = roster.sample_opponents(3)

    assert len({entry.agent_id for entry in sampled}) == 3


def test_avg_is_excluded_until_added() -> None:
    roster = _make_roster()

    sampled = roster.sample_opponents(10)

    assert "avg" not in {entry.kind for entry in sampled}


def test_checkpoint_admission_adds_live_draw_prior() -> None:
    roster = _make_roster()

    entry = roster.add_checkpoint("/ckpt/a.pt", global_step=20, update=4)

    checkpoint_index = roster.counts.index(entry.agent_id)
    live_index = roster.counts.index("live")
    assert roster.counts.tensor[checkpoint_index, live_index, 2] == 10.0


def test_crowding_evicts_oldest_member_of_closest_unprotected_pair() -> None:
    roster = _make_roster(league_size=6)
    elos = (0.0, 10.0, 1_000.0, 2_000.0, 3_000.0, 4_000.0, 5_000.0)
    for update, elo in enumerate(elos, start=1):
        roster.add_checkpoint(
            f"/ckpt/{update}.pt",
            global_step=update,
            update=update,
            initial_elo=elo,
        )

    checkpoint_ids = {entry.agent_id for entry in roster.entries if entry.kind == "checkpoint"}

    assert "ckpt_1" not in checkpoint_ids


def test_retention_removes_evicted_agent_counts() -> None:
    roster = _make_roster(league_size=1)
    roster.add_checkpoint("/ckpt/one.pt", global_step=1, update=1)
    roster.add_checkpoint("/ckpt/two.pt", global_step=2, update=2)

    checkpoint_ids = {entry.agent_id for entry in roster.entries if entry.kind == "checkpoint"}

    assert set(roster.counts.agent_ids) == {"live", "random", "scripted", *checkpoint_ids}


def test_save_load_v2_round_trip_includes_counts(tmp_path) -> None:
    roster = _make_roster()
    roster.add_avg(global_step=10, update=2)
    checkpoint = roster.add_checkpoint("/ckpt/a.pt", global_step=20, update=4)
    roster.counts.add_pair("live", checkpoint.agent_id, wins=2.5, losses=1.0)
    path = tmp_path / "roster.json"
    roster.save_json(path)

    restored = _make_roster()
    restored.load_json(path)

    assert restored.entries == roster.entries
    assert torch.equal(restored.counts.tensor, roster.counts.tensor)


def test_loader_rejects_v1_roster(tmp_path) -> None:
    path = tmp_path / "roster.json"
    path.write_text(json.dumps({"entries": []}))

    with pytest.raises(ValueError, match="version 2"):
        _make_roster().load_json(path)


def test_kept_paths_returns_retained_checkpoint_paths_only() -> None:
    roster = _make_roster()
    roster.add_avg(global_step=1, update=1)
    roster.add_checkpoint("/ckpt/a.pt", global_step=2, update=2)
    roster.add_checkpoint("/ckpt/b.pt", global_step=3, update=3)

    assert roster.kept_paths() == {"/ckpt/a.pt", "/ckpt/b.pt"}
