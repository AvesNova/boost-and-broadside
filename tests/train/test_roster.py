"""Tests for the v3 league roster, PFSP curriculum, and retention policy."""

import json
import math

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


def test_loader_rejects_older_roster_versions(tmp_path) -> None:
    path = tmp_path / "roster.json"
    path.write_text(json.dumps({"version": 2, "entries": []}))

    with pytest.raises(ValueError, match="version 3"):
        _make_roster().load_json(path)


def test_kept_paths_returns_retained_checkpoint_paths_only() -> None:
    roster = _make_roster()
    roster.add_avg(global_step=1, update=1)
    roster.add_checkpoint("/ckpt/a.pt", global_step=2, update=2)
    roster.add_checkpoint("/ckpt/b.pt", global_step=3, update=3)

    assert roster.kept_paths() == {"/ckpt/a.pt", "/ckpt/b.pt"}


def test_record_outcomes_writes_lifetime_counts_and_update_tally() -> None:
    roster = _make_roster()
    live = roster.counts.index("live")
    random_index = roster.counts.index("random")

    roster.record_outcomes(
        torch.tensor([live, live]),
        torch.tensor([random_index, random_index]),
        torch.tensor([0, 1]),
    )

    assert roster.counts.tensor[live, random_index].tolist() == [1.0, 1.0, 0.0]
    assert roster.update_tally.tensor[live, random_index].tolist() == [1.0, 1.0, 0.0]


def test_reset_update_tally_preserves_lifetime_counts() -> None:
    roster = _make_roster()
    live = roster.counts.index("live")
    random_index = roster.counts.index("random")
    roster.record_outcomes(torch.tensor([live]), torch.tensor([random_index]), torch.tensor([0]))

    roster.reset_update_tally()

    assert roster.update_tally.tensor.sum() == 0.0
    assert roster.counts.tensor.sum() == 1.0


def test_layout_version_bumps_and_tally_layout_tracks_counts(tmp_path) -> None:
    roster = _make_roster(league_size=1)
    version = roster.layout_version

    roster.add_avg(global_step=1, update=1)
    assert roster.layout_version > version
    assert roster.update_tally.agent_ids == roster.counts.agent_ids

    version = roster.layout_version
    roster.add_checkpoint(str(tmp_path / "a.pt"), global_step=2, update=2)
    roster.add_checkpoint(str(tmp_path / "b.pt"), global_step=3, update=3)  # evicts one
    assert roster.layout_version > version
    assert roster.update_tally.agent_ids == roster.counts.agent_ids


def test_scripted_starts_at_1000_and_holds_until_evidence() -> None:
    roster = _make_roster()
    assert roster.ratings["scripted"] == 1_000.0
    assert roster.entry("scripted").elo == 1_000.0

    live = roster.counts.index("live")
    random_index = roster.counts.index("random")
    roster.record_outcomes(
        torch.tensor([live] * 4), torch.tensor([random_index] * 4), torch.tensor([0, 0, 0, 1])
    )
    ratings, _ = roster.solve_ratings()

    assert ratings["scripted"] == 1_000.0


def test_protected_anchor_survives_eviction_pressure() -> None:
    roster = _make_roster(league_size=1)
    roster.add_checkpoint(
        "/ckpt/anchor.pt", global_step=0, update=0, initial_elo=0.0, protected=True
    )
    roster.add_checkpoint("/ckpt/one.pt", global_step=1, update=1)
    roster.add_checkpoint("/ckpt/two.pt", global_step=2, update=2)

    checkpoint_ids = {entry.agent_id for entry in roster.entries if entry.kind == "checkpoint"}

    # The anchor never counts against league_size and is never evicted.
    assert "ckpt_0" in checkpoint_ids
    assert len(checkpoint_ids) == 2
    assert roster.anchor_id() == "ckpt_0"


def test_pool_and_probe_partition_helpers() -> None:
    roster = _make_roster()
    roster.add_avg(global_step=1, update=1)
    roster.add_checkpoint("/ckpt/a.pt", global_step=2, update=2)

    assert set(roster.pool_agent_ids()) == {"live", "avg", "ckpt_2"}
    assert set(roster.probe_agent_ids()) == {"random", "scripted"}
    assert roster.anchor_id() is None
    assert roster.newest_checkpoint().agent_id == "ckpt_2"


def test_solve_ratings_routes_pool_and_probe_stages() -> None:
    roster = _make_roster()
    roster.add_checkpoint(
        "/ckpt/anchor.pt", global_step=0, update=0, initial_elo=0.0, protected=True
    )
    roster.counts.add_pair("live", "ckpt_0", wins=600.0, losses=400.0)
    baseline, _ = roster.solve_ratings()
    live_baseline = baseline["live"]

    # Heavy scripted stomps against live: probe rating moves, pool does not.
    roster.counts.add_pair("live", "scripted", losses=5000.0)
    ratings, errors = roster.solve_ratings()

    assert ratings["live"] == pytest.approx(live_baseline)
    assert ratings["ckpt_0"] == 0.0
    assert ratings["scripted"] > ratings["live"]
    assert math.isfinite(errors["scripted"])


def test_gap_triggered_admission_logic() -> None:
    from types import SimpleNamespace

    from boost_and_broadside.train.rl.checkpoint import CheckpointMixin

    roster = _make_roster()
    roster.add_checkpoint(
        "/ckpt/anchor.pt", global_step=0, update=0, initial_elo=0.0, protected=True
    )
    stub = SimpleNamespace(
        cfg=SimpleNamespace(
            admission_min_interval=3,
            league_admission_interval=25,
            admission_winrate_trigger=0.65,
        ),
        roster=roster,
        _policy_gradient_coef=1.0,
        _last_admission_update=0,
    )

    # Unidentified ratings: only the interval fallback fires.
    roster.standard_errors = {"live": math.inf, "ckpt_0": 0.0}
    assert not CheckpointMixin._admission_due(stub, 10)
    assert CheckpointMixin._admission_due(stub, 25)

    # Identified and live well above the newest rung: gap trigger fires.
    roster.refresh_ratings({**roster.ratings, "live": 200.0})
    roster.standard_errors = {"live": 10.0, "ckpt_0": 0.0}
    assert CheckpointMixin._admission_due(stub, 10)
    # ...but never within the minimum spacing of the last admission.
    stub._last_admission_update = 9
    assert not CheckpointMixin._admission_due(stub, 10)
