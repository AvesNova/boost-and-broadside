"""Crossover fights (`bnb.league.crossover`, `bnb crossover`): the match engine plays each
pair of active ship counts with the first player's count on its own side whatever the
seat, the scripted agent against itself scores about one half at equal counts and wins
with more ships, a network plays, and the interval, the pairs' syntax and the crossing
are as documented; on a small 2v2 or 3v3 game on the CPU."""

import dataclasses
import json
import math

import pytest
import torch

from bnb.contracts import GameConfig
from bnb.contracts.obs import SHIP
from bnb.game import api
from bnb.game.state import benched_of
from bnb.league import crossover as fights
from bnb.league.matches import Arena, ArenaConfig, Player, player
from bnb.runs import checkpoint
from bnb.runtime.opponents import Rung
from tests.runs.runs_cases import tiny_model

pytestmark = pytest.mark.usefixtures("global_generator")

CPU = torch.device("cpu")
# Short, small and lethal enough that most games end by the win rule (one kill per ship).
GAME = GameConfig(3, 2, 500.0, episode_seconds=20.0, respawn_seconds=1.0, kills_per_ship=1)
LAYOUT = ArenaConfig(blocks=2, block_environments=8, length=16, compiled=False)


def _scripted() -> Player:
    return Player("scripted", rung=Rung(1.0))


def test_each_environment_gives_the_first_player_its_count_whatever_its_seat():
    arena = Arena([_scripted(), _scripted()], GAME, CPU, LAYOUT, benching=True)
    active = arena._active([(3, 1), (2, 3)])  # (B, 2) by team
    first_team = arena.seats.long()
    first = active.gather(1, first_team[:, None])[:, 0]
    second = active.gather(1, 1 - first_team[:, None])[:, 0]
    rows = LAYOUT.block_environments
    assert first.tolist() == [3] * rows + [2] * rows
    assert second.tolist() == [1] * rows + [3] * rows
    assert set(first_team.tolist()) == {0, 1}  # both seats in every block
    with pytest.raises(ValueError, match="from 1 to"):
        arena.play([(0, 1), (0, 1)], [(4, 3), (3, 3)])


def test_the_rollout_shows_the_benched_flag_from_its_first_observation_on():
    """The restart's observation (before any tick) and every one after it flag exactly the
    benched rows of both views, as the rollout's active counts give them."""
    arena = Arena([_scripted(), _scripted()], GAME, CPU, LAYOUT, benching=True)
    rollout = arena.rollout
    rollout.restart(arena._active([(3, 1), (2, 3)]))
    order = api.view_order(GAME.ships_per_team, CPU)
    rows = benched_of(GAME.ships_per_team, rollout.carry["active"])[:, order].float()
    assert rows.any()
    assert torch.equal(rollout.carry["obs"]["ships"][..., SHIP["benched"]], rows)
    rollout.run()
    assert torch.equal(rollout.carry["obs"]["ships"][..., SHIP["benched"]], rows)


def test_an_engine_without_benching_refuses_counts():
    arena = Arena([_scripted(), _scripted()], GAME, CPU, LAYOUT)
    with pytest.raises(ValueError, match="benching"):
        arena.play([(0, 1), (0, 1)], [(3, 3), (3, 3)])


def test_the_scripted_agent_against_itself_scores_a_half_at_equal_counts_and_wins_with_more():
    rows = fights.crossover(_scripted(), _scripted(), [(3, 3), (3, 1)], 16, GAME, CPU, LAYOUT)
    equal, ahead = rows
    for row in rows:
        assert row["games"] == row["wins"] + row["draws"] + row["losses"] == 16
        assert row["first_identity"] == row["second_identity"] == "scripted"
        assert row["score_low"] <= row["score"] <= row["score_high"]
    # one agent on both sides, the seats split: a fair coin, inside its 95% interval
    assert equal["score_low"] < 0.5 < equal["score_high"]
    assert (equal["handicap"], ahead["handicap"]) == (0, -2)
    assert ahead["score"] > 0.9  # three ships against one


@pytest.mark.parametrize("name", ["relational", "yemong"])
def test_a_network_plays_crossover_pairs(tmp_path, name):
    """A Yemong plays with its benched mask on (`Yemong.mask_benched`)."""
    game = GameConfig(2, 2, 700.0, episode_seconds=4.0, respawn_seconds=1.0)
    directory = tmp_path / "policy"
    checkpoint.save_policy(directory, tiny_model(name, seed=1, game=game))
    network = player(str(directory))
    layout = dataclasses.replace(LAYOUT, block_environments=4)
    rows = fights.crossover(network, _scripted(), [(2, 2), (1, 2), (2, 1)], 8, game, CPU, layout)
    assert [(row["first_active"], row["second_active"]) for row in rows] == [(2, 2), (1, 2), (2, 1)]
    assert all(row["games"] == 8 for row in rows)
    assert rows[0]["first_identity"].startswith("sha256:")
    assert rows[0]["second_identity"] == "scripted"


def test_the_counts_must_fit_the_game_and_the_games_be_even():
    with pytest.raises(ValueError, match="ships per team"):
        fights.crossover(_scripted(), _scripted(), [(4, 3)], 4, GAME, CPU, LAYOUT)
    with pytest.raises(ValueError, match="even"):
        fights.crossover(_scripted(), _scripted(), [(3, 3)], 5, GAME, CPU, LAYOUT)


def test_a_pair_is_two_positive_counts():
    assert fights.parse_pair("5:8") == (5, 8)
    for bad in ("5", "5:", "a:3", "0:3", "5-8"):
        with pytest.raises(ValueError, match="a:b"):
            fights.parse_pair(bad)


def test_the_wilson_interval_is_the_textbook_one():
    # 8 of 10: Wilson's 95% interval is (0.4902, 0.9433)
    low, high = fights.wilson_interval(0.8, 10)
    assert low == pytest.approx(0.4902, abs=1e-4) and high == pytest.approx(0.9433, abs=1e-4)
    assert fights.wilson_interval(0.0, 0) == (0.0, 1.0)
    low, high = fights.wilson_interval(1.0, 20)
    assert high == 1.0 and 0.8 < low < 0.85


def test_the_crossing_is_interpolated_between_neighbouring_pairs():
    def row(first: int, second: int, score: float) -> dict:
        return dict(first_active=first, second_active=second, handicap=second - first, score=score)

    rows = [row(5, 5, 0.8), row(5, 6, 0.6), row(5, 7, 0.3), row(5, 8, 0.1)]
    (found,) = fights.crossings(rows)
    assert found["between"] == ["5:6", "5:7"]
    assert found["second_active"] == pytest.approx(6 + 1 / 3)
    assert found["handicap"] == pytest.approx(1 + 1 / 3) and found["first_active"] == 5
    exact = fights.crossings([row(5, 5, 0.7), row(4, 5, 0.5), row(3, 5, 0.2)])
    assert [crossing["between"] for crossing in exact] == [["4:5"]]
    assert fights.crossings([row(5, 5, 0.7), row(4, 5, 0.6)]) == []
    assert math.isnan(fights.summary(torch.tensor([0, 0, 0]))["score"])


def test_the_command_parses_its_options():
    import tyro

    from bnb.cli import COMMANDS, Crossover

    options = tyro.cli(
        Crossover,
        args=["policy", "--pairs", "5:5", "5:6", "--ships-per-team", "8", "--games", "64"],
    )
    assert options.players == ("policy",) and options.pairs == ("5:5", "5:6")
    assert options.ships_per_team == 8 and options.games == 64 and options.preset == "deathmatch"
    assert "crossover" in COMMANDS


def test_the_command_prints_the_table_and_records_the_rows(tmp_path, monkeypatch, capsys):
    import functools

    from bnb import cli
    from bnb.game import api
    from bnb.league import matches

    monkeypatch.setitem(api.PRESETS, "tiny_3v3", GAME)
    monkeypatch.setattr(
        matches, "ArenaConfig", functools.partial(ArenaConfig, length=16, compiled=False)
    )
    options = cli.Crossover(
        players=("scripted",),
        pairs=("3:3", "3:1"),
        preset="tiny_3v3",
        games=4,
        blocks=2,
        block_environments=4,
        device="cpu",
        out=str(tmp_path),
    )
    cli.crossover(options)
    printed = capsys.readouterr().out
    assert "scripted against scripted, tiny_3v3 at 3v3, 4 games per pair" in printed
    (directory,) = (tmp_path / "artifacts" / "crossover").iterdir()
    lines = (directory / "pairs.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    assert [(row["first_active"], row["second_active"]) for row in rows] == [(3, 3), (3, 1)]
    result = json.loads((directory / "result.json").read_text())
    assert result["payload"]["rows"] == rows and result["payload"]["seed"] == 0
    assert "command" in result["provenance"]


def test_an_engine_with_benching_masks_a_yemongs_benched_ships(tmp_path):
    """The benching engine turns `Yemong.mask_benched` on; the full-team one leaves it off."""
    game = GameConfig(2, 2, 700.0, episode_seconds=4.0, respawn_seconds=1.0)
    directory = tmp_path / "policy"
    checkpoint.save_policy(directory, tiny_model("yemong", seed=1, game=game))
    layout = dataclasses.replace(LAYOUT, block_environments=4)
    for benching in (False, True):
        network = player(str(directory))
        arena = Arena([network, _scripted()], game, CPU, layout, benching=benching)
        assert arena.groups[0].mask_benched is benching


def test_the_command_flies_its_lineup_for_both_players(monkeypatch, tmp_path):
    from bnb import cli

    seen = {}

    def fake(first, second, pairs, games, game, device, layout):
        seen["types"] = (first.ship_type, second.ship_type)
        raise ValueError("stop")

    monkeypatch.setattr(fights, "crossover", fake)
    lineup = ("Yongma", "Yongma", "Jujak", "Jujak", "Jujak")
    options = cli.Crossover(
        players=("scripted",), pairs=("5:4",), preset="real_scale_deathmatch", lineup=lineup
    )
    with pytest.raises(SystemExit, match="stop"):
        cli.crossover(options)
    assert seen["types"] == ((0, 0, 1, 1, 1), (0, 0, 1, 1, 1))


def test_head_to_head_flies_its_lineup_for_every_player(monkeypatch):
    from bnb import cli
    from bnb.league import tournament

    seen = {}

    def fake(players, games, presets, device, layout, deathmatch_score):
        seen["types"] = [one.ship_type for one in players]
        seen["layout"] = (layout.blocks, layout.block_environments)
        raise ValueError("stop")

    monkeypatch.setattr(tournament, "head_to_head", fake)
    lineup = ("Yongma", "Yongma", "Jujak", "Jujak", "Jujak")
    options = cli.HeadToHead(
        players=("scripted", "random"),
        presets=("real_scale_deathmatch",),
        lineup=lineup,
        blocks=2,
        block_environments=96,
    )
    with pytest.raises(SystemExit, match="stop"):
        cli.head_to_head(options)
    assert seen["types"] == [(0, 0, 1, 1, 1)] * 2 and seen["layout"] == (2, 96)
