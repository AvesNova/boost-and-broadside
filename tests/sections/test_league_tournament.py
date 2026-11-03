"""Published ratings and head-to-head (league spec 6): the match engine, the tournament's
anchor and stopping rule, and head-to-head's report, on a small 2v2 game on the CPU."""

import math

import pytest
import torch

from bnb.contracts import GameConfig
from bnb.league import tournament
from bnb.league.matches import Player, ladder_probability, play, player
from bnb.models.relational import Relational, RelationalBuild, RelationalConfig
from bnb.runs import checkpoint

pytestmark = pytest.mark.usefixtures("global_generator")

GAME = GameConfig(2, 2, 1500.0, episode_seconds=2.0, respawn_seconds=1.0)
CPU = torch.device("cpu")
SCRIPTED = Player("scripted", scripted_probability=1.0)
RANDOM = Player("random", scripted_probability=0.0)
HALF = Player("rung_0.5", scripted_probability=0.5)


def test_ladder_names_parse_and_others_do_not():
    assert ladder_probability("random") == 0.0 and ladder_probability("scripted") == 1.0
    assert ladder_probability("rung_0.3") == 0.3
    assert ladder_probability("rung_1.5") is None and ladder_probability("rung_x") is None
    with pytest.raises(ValueError, match="not a policy file"):
        player("no-such-player")


def test_every_pair_plays_its_games_once_each():
    results = play([SCRIPTED, RANDOM, HALF], [(0, 1, 8), (2, 1, 4)], GAME, CPU, length=32)
    assert results.sum(1).tolist() == [8, 4]


def test_a_pair_needs_an_even_number_of_games():
    with pytest.raises(ValueError, match="split evenly"):
        play([SCRIPTED, RANDOM], [(0, 1, 3)], GAME, CPU)


def test_a_policy_file_plays(tmp_path):
    model = Relational(RelationalBuild(RelationalConfig(hidden=16), 2, 4))
    checkpoint.save_policy(tmp_path / "policy", model)
    network = player(str(tmp_path / "policy"))
    results = play([network, RANDOM], [(0, 1, 4)], GAME, CPU, length=32)
    assert int(results.sum()) == 4


def test_the_tournament_pins_the_scripted_agent_at_1000_and_stops_on_its_budget():
    logged = []
    config = tournament.TournamentConfig(first_games=4, batch_games=12, budget=24, length=32)
    rows = tournament.rate([RANDOM, SCRIPTED, HALF], "small", CPU, config, logged.append, GAME)
    by_name = {row["player"]: row for row in rows}
    assert by_name["scripted"]["calibrated_elo"] == 1000.0 and by_name["scripted"]["stderr"] == 0.0
    assert by_name["random"]["stderr"] > 0.0  # random is fitted too
    played = sum(row["games"] for row in rows) // 2  # each game counts for both players
    # a pair's games round up to an even count, so the budget can be passed by one per pair
    assert 24 <= played <= 24 + 3 and len(logged) == 2
    assert {row["map"] for row in rows} == {"small"}
    assert all(row["fleet_size"] == 2 and row["mode"] == GAME.mode for row in rows)


def test_the_tournament_needs_the_scripted_agent():
    with pytest.raises(ValueError, match="scripted agent"):
        tournament.rate([RANDOM, HALF], "small", CPU, game=GAME)


def test_head_to_head_reports_each_pair_in_total_and_per_preset():
    rows = tournament.head_to_head(
        [SCRIPTED, RANDOM, HALF, RANDOM], 6, ("one", "two"), CPU, 32, {"one": GAME, "two": GAME}
    )
    assert [(row["first"], row["second"]) for row in rows] == [
        ("scripted", "random"),
        ("rung_0.5", "random"),
    ]
    for row in rows:
        assert row["wins"] + row["draws"] + row["losses"] == 12
        assert sum(part["wins"] for part in row["presets"].values()) == row["wins"]
        assert 0.0 <= row["score"] <= 1.0 and math.isfinite(row["stderr"])


def test_head_to_head_takes_pairs_and_an_even_game_count():
    with pytest.raises(ValueError, match="pairs"):
        tournament.head_to_head([SCRIPTED, RANDOM, HALF], 4, ("one",), CPU, games_of={"one": GAME})
    with pytest.raises(ValueError, match="even"):
        tournament.head_to_head([SCRIPTED, RANDOM], 3, ("one",), CPU, games_of={"one": GAME})


def test_the_score_error_is_the_binomial_one_without_draws():
    summary = tournament._summary(torch.tensor([30, 0, 10]))
    assert summary["score"] == 0.75
    assert summary["stderr"] == pytest.approx(math.sqrt(0.75 * 0.25 / 40))


def test_the_commands_parse_their_options():
    import tyro

    from bnb.cli import HeadToHead, Ratings

    pairs = tyro.cli(HeadToHead, args=["a", "b", "scripted", "random", "--games", "8"])
    assert pairs.players == ("a", "b", "scripted", "random") and pairs.games == 8
    rated = tyro.cli(Ratings, args=["run", "--with", "x", "y", "--target-stderr", "30"])
    assert rated.run == "run" and rated.with_ == ("x", "y") and rated.target_stderr == 30.0
