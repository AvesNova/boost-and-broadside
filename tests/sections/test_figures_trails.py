"""The trail stills (`bnb.figures.trails`) and the match record they read
(`bnb.figures.match`): a hand-built record's windows, lives and markers, a short Imugi
against Imugi match drawn end to end with its provenance, and a tiny Yemong policy's
belief ghosts kept in the record."""

import dataclasses
import math

import numpy as np
import pytest
import torch
from play_cases import SMALL, tiny_policy

from bnb.contracts.config import ELIMINATION
from bnb.figures import match, trails

TICK = 1.0 / 30.0


def hand_built(ticks: int = 60, ships: int = 2) -> dict:
    """A record of two ships flying +x at 3 m a tick, ship 0 hit on ticks 40 to 44 and
    killed on 45, then respawned at the far side on 50; ship 1 hidden from team 0 from tick
    20, with team 0's ghost of it every 10 ticks."""
    x = np.tile(3.0 * np.arange(ticks, dtype=np.float32)[:, None], (1, ships))
    y = np.zeros((ticks, ships), np.float32)
    y[:, 1] = 200.0
    alive = np.ones((ticks, ships), bool)
    alive[46:50, 0] = False
    x[50:, 0] += 2000.0  # the respawn: a teleport
    visible = np.ones((ticks, 2, ships), bool)
    visible[20:, 0, 1] = False
    hit = np.zeros((ticks, ships), bool)
    hit[40:45, 0] = True
    died = np.zeros((ticks, ships), bool)
    died[45, 0] = True
    ghost = np.full((ticks, 2, ships), np.nan, np.float32)
    ghost[::10, 0, 1] = x[::10, 1] + 5.0
    sigma = np.where(np.isnan(ghost), np.nan, 30.0).astype(np.float32)
    fired = np.zeros((ticks, ships), bool)
    fired[::3] = True
    record = dict(
        x=x,
        y=y,
        heading=np.zeros((ticks, ships), np.float32),
        alive=alive,
        visible=visible,
        team=np.array([0, 1]),
        fired=fired,
        hit=hit,
        died=died,
        done=np.arange(ticks) == ticks - 1,
        clock=TICK * np.arange(ticks, dtype=np.float32),
        boundary_radius=np.full(ticks, 3000.0, np.float32),
        fogged=True,
        arena_radius=3000.0,
        fields=dict(
            x=np.array([0.0, 500.0]),
            y=np.array([0.0, 0.0]),
            radius=np.array([100.0, 80.0]),
            level=np.array([1, -1]),
        ),
        ghost_x=ghost,
        ghost_y=np.where(np.isnan(ghost), np.nan, 205.0).astype(np.float32),
        ghost_sigma=sigma,
        about=dict(
            preset="test", ships_per_team=1, tick_length=TICK, labels=["one", "two"], seed=0
        ),
    )
    return dict(meta=dict(policy="hand built"), match=match._tensors(record))


def test_a_life_ends_at_death_and_at_a_respawn_s_teleport():
    record = hand_built()["match"]
    lives = trails._lives(
        record["alive"][:, 0].numpy(), *(record[k][:, 0].numpy() for k in "xy"), 0, 59
    )
    assert lives == [(0, 45), (50, 59)]
    assert trails._lives(
        record["alive"][:, 0].numpy(), *(record[k][:, 0].numpy() for k in "xy"), 30, 47
    ) == [(30, 45)]


def test_the_window_is_in_seconds_and_the_busiest_finds_the_hits():
    record = hand_built()["match"]
    assert trails.window_ticks(record, trails.StillConfig(start=1.0, end=1.5)) == (30, 45)
    assert trails.window_ticks(record, trails.StillConfig()) == (0, 59)
    with pytest.raises(ValueError):
        trails.window_ticks(record, trails.StillConfig(start=1.5, end=1.0))
    start, end = trails.busiest_window(record, TICK, 0.3)  # 9 ticks
    assert round(start / TICK) <= 40 and round(end / TICK) >= 45


def test_each_team_takes_its_own_ramp_light_to_dark_by_slot():
    colours = trails.ship_colours(np.array([0, 0, 0, 1, 1, 1]), "light")
    assert colours[:3] == [trails.RAMPS[0][0][i] for i in (0, 2, 4)]
    assert colours[3:] == [trails.RAMPS[1][0][i] for i in (0, 2, 4)]
    assert trails.ship_colours(np.array([0, 1]), "dark") == [
        trails.RAMPS[0][1][2],
        trails.RAMPS[1][1][2],
    ]


def test_a_still_counts_what_it_draws_and_writes_every_format(tmp_path):
    pytest.importorskip("matplotlib")
    from PIL import Image

    artifact = hand_built()
    config = trails.StillConfig(
        fog=True, shots=True, hits=True, ends=True, ghost_seconds=0.3, formats=("png", "svg")
    )
    paths, numbers = trails.still(artifact, tmp_path, "still", config)
    assert [path.suffix for path in paths] == [".png", ".svg"]
    assert numbers["deaths"] == [1, 0] and numbers["hits"] == [5, 0]
    # a ghost every 10 ticks while hidden (ticks 20 to 50), at most one per 9 ticks drawn
    assert numbers["ghosts"] == 4
    assert Image.open(paths[0]).text["Source"] == "hand built"
    _, numbers = trails.still(artifact, tmp_path, "quiet", trails.StillConfig(theme="dark"))
    assert numbers["shots"] == [0, 0] and numbers["hits"] == [0, 0] and numbers["ghosts"] == 0
    assert numbers["deaths"] == [1, 0]  # the X is always drawn


def test_a_fleet_still_refuses_every_mark_at_once():
    config = trails.StillConfig(shots=True, hits=True, fog=True)
    trails.check_marks(config, 1)  # one ship a side: allowed
    trails.check_marks(dataclasses.replace(config, hits=False), 5)
    with pytest.raises(ValueError):
        trails.check_marks(config, 5)


def test_an_imugi_match_records_every_tick_and_draws_the_trails_set(tmp_path):
    pytest.importorskip("matplotlib")
    from PIL import Image

    from bnb.figures.sets import FIGURE_SETS, drawable

    game = dataclasses.replace(SMALL, ships_per_team=1)
    config = match.MatchConfig(seconds=1.0, ghost_ticks=10)
    artifact = match.record(("imugi", "imugi"), config, game=game)
    record = artifact["match"]
    ticks = round(1.0 / record["about"]["tick_length"])
    assert record["x"].shape == (ticks, 2) and record["visible"].shape == (ticks, 2, 2)
    assert record["ghost_x"].shape == (ticks, 2, 2)
    assert record["team"].tolist() == [0, 1]
    assert drawable("trails", artifact) and not drawable("belief", artifact)
    written, numbers = FIGURE_SETS["trails"](artifact, tmp_path)
    assert {path.stem for path in written} >= {"trails_game", "trails_game_dark", "trails_busiest"}
    assert "Commit" in Image.open(written[0]).text


def test_a_policy_s_ghosts_carry_its_belief_sigma(tmp_path):
    game = dataclasses.replace(SMALL, ships_per_team=1, mode=ELIMINATION, arena_radius=3000.0)
    policy = tiny_policy(tmp_path / "policy", game, name="yemong")
    config = match.MatchConfig(seconds=0.5, ghost_ticks=5, fogged=True, opponent="policy")
    artifact = match.record(match.sides_of(policy, config), config, game=game)
    record = artifact["match"]
    sampled = ~torch.isnan(record["ghost_x"])
    off_sample = torch.arange(sampled.shape[0]) % 5 != 0
    assert not sampled[off_sample].any()
    sigma = record["ghost_sigma"][sampled]
    assert torch.isfinite(sigma).all() and (sigma > 0).all()
    assert all(math.isfinite(float(v)) for v in record["clock"])
