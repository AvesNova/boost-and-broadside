"""The shared rating pool (league spec 6, `bnb.league.rating_pool`): stable names, one file
per writer merged on read, the import of `bnb ratings` records, a rating session over the
pool, and the goodness-of-fit report (`bnb.league.goodness_of_fit`), on the CPU with
scripted players and a Bradley-Terry model of the games."""

import json
import math

import pytest
import torch

from bnb.league import goodness_of_fit, rating_pool, tournament
from bnb.league.design import fit_posterior
from bnb.league.matches import ArenaConfig, player
from bnb.league.rating_pool import PoolPlayer, RatingPool
from bnb.league.ratings import win_probability

DTYPE = torch.float64
SCORING = ("win_loss", "win_loss")
LADDER = ["scripted", "rung_T2", "rung_T4", "rung_T8", "random"]


def test_a_policy_is_named_by_its_run_steps_and_kind():
    assert rating_pool.policy_name("/x/runs/rl-v20/policies/step_75497472/exponential_average") == (
        "rl-v20/step_75497472/exponential_average",
        "rl-v20",
        75497472,
        "exponential_average",
    )
    assert rating_pool.policy_name("r/league/step_5/uniform_average/")[0] == (
        "r/step_5/uniform_average"
    )
    assert rating_pool.policy_name("/somewhere/policy") is None


def test_a_pool_is_created_once_and_holds_its_conditions(tmp_path):
    pool = RatingPool.create(tmp_path / "pool", ("A", "B"), 1.5, "win_loss")
    assert pool.settings["lineup"] == ["A", "B"] and pool.settings["arena_scale"] == 1.5
    assert "scripted" in pool.players()[0]
    again = RatingPool.create(tmp_path / "pool", ("A", "B"), 1.5, "win_loss")
    assert list(again.players()[0]) == ["scripted"]  # the anchor registered once
    with pytest.raises(ValueError, match="other conditions"):
        RatingPool.create(tmp_path / "pool", ("A",), 1.5, "win_loss")
    with pytest.raises(FileNotFoundError, match="not a rating pool"):
        RatingPool(tmp_path / "elsewhere")


def test_writers_keep_their_own_files_and_a_reader_merges_them(tmp_path):
    RatingPool.create(tmp_path)
    first, second = RatingPool(tmp_path, "machine-a"), RatingPool(tmp_path, "machine-b")
    first.register([PoolPlayer("r/step_1/exponential_average", "sha256:1", "r", 1, "e")])
    second.register([PoolPlayer("r/step_2/exponential_average", "sha256:2", "r", 2, "e")])
    batch = dict(game_version=1, matchups=[["scripted", "r/step_1/exponential_average"]])
    first.append_batch("frontline", dict(batch, id="one"))
    second.append_batch("frontline", dict(batch, id="two"))
    second.append_batch("frontline", dict(batch, id="one"))  # a copy counts once
    files = sorted(path.name for path in (tmp_path / "games" / "frontline").iterdir())
    assert files == ["machine-a.jsonl", "machine-b.jsonl"]
    players, conflicts = RatingPool(tmp_path).players()
    assert set(players) == {
        "scripted",
        "r/step_1/exponential_average",
        "r/step_2/exponential_average",
    }
    assert not conflicts
    batches, damaged = RatingPool(tmp_path).batches("frontline")
    assert sorted(entry["id"] for entry in batches) == ["one", "two"] and damaged == 0
    # a line in progress (no newline yet) is skipped quietly; a damaged whole line is counted
    with (tmp_path / "games" / "frontline" / "machine-c.jsonl").open("w") as file:
        file.write('{"id": "three"}\n{not json}\n{"id": "fo')
    batches, damaged = RatingPool(tmp_path).batches("frontline")
    assert sorted(entry["id"] for entry in batches) == ["one", "three", "two"] and damaged == 1


def test_a_name_with_other_weights_is_refused_and_its_games_not_counted(tmp_path):
    pool = RatingPool.create(tmp_path)
    pool.register([PoolPlayer("r/step_1/e", "sha256:1", "r", 1, "e", ("/a",))])
    pool.register([PoolPlayer("r/step_1/e", "sha256:1", "r", 1, "e", ("/b",))])
    assert pool.players()[0]["r/step_1/e"].paths == ("/a", "/b")
    with pytest.raises(ValueError, match="other weights"):
        pool.register([PoolPlayer("r/step_1/e", "sha256:9", "r", 1, "e")])
    batch = dict(
        id="x",
        game_version=1,
        matchups=[["scripted", "r/step_1/e"], ["scripted", "r/step_1/e"]],
        results=[[3, 0, 5], [4, 0, 4]],
        scoring="win_loss",
    )
    pool.append_batch(
        "p", dict(batch, identities={"scripted": "scripted", "r/step_1/e": "sha256:1"})
    )
    pool.append_batch(
        "p", dict(batch, id="y", identities={"scripted": "scripted", "r/step_1/e": "sha256:9"})
    )
    from bnb.game import VERSION

    batches = [dict(entry, game_version=VERSION) for entry in pool.batches("p")[0]]
    counted = rating_pool.tally(batches, pool.players()[0], SCORING)
    assert counted.games == 16 and counted.skipped == {"other weights than the registered ones": 2}
    assert float(counted.wins[0, 1]) == 7.0


def _bradley_terry(truth: dict[str, float], block_games: int, seed: int):
    """play_batch for the pool: each block's first player wins each game with its model
    probability (no draws), recorded with the full outcome's columns."""
    generator = torch.Generator().manual_seed(seed)

    def play(field, matchups):
        rows = []
        for first, second in matchups:
            probability = win_probability(
                torch.tensor(truth[field[first].name], dtype=DTYPE),
                torch.tensor(truth[field[second].name], dtype=DTYPE),
            )
            won = int(
                torch.binomial(
                    torch.tensor(float(block_games), dtype=DTYPE), probability, generator
                )
            )
            rows.append([won, 0, block_games - won])
        shares = [float(row[0]) for row in rows]
        kills = [[row[0], row[2]] for row in rows]
        return torch.tensor(rows), dict(kill_share=shares, kills=kills)

    return play


def _ladder_pool(tmp_path) -> RatingPool:
    pool = RatingPool.create(tmp_path)
    pool.register([PoolPlayer(name, name) for name in LADDER[1:]])
    return pool


def test_a_session_rates_the_pool_and_a_new_player_joins_it(tmp_path):
    truth = {"scripted": 0.0, "rung_T2": -150.0, "rung_T4": -300.0, "rung_T8": -420.0}
    truth["random"] = -700.0
    pool = _ladder_pool(tmp_path)
    config = tournament.TournamentConfig(
        ArenaConfig(blocks=4, block_environments=32), target_stderr=60.0, budget=128 * 30
    )
    logged = []
    batches = rating_pool.rate(
        pool,
        "frontline",
        lambda one: player(one.name),
        _bradley_terry(truth, 32, 0),
        config,
        SCORING,
        log=logged.append,
        extra=dict(seed=7),
    )
    assert batches > 0 and "every target of the session is met" in logged[-1]
    rows, check, notes = rating_pool.table(pool, "frontline", SCORING, 1000.0)
    by_name = {row["player"]: row for row in rows}
    assert by_name["scripted"]["calibrated_elo"] == 1000.0
    for name, rating in truth.items():
        assert abs(by_name[name]["calibrated_elo"] - 1000.0 - rating) <= 4 * max(
            by_name[name]["stderr"], 1.0
        )
    assert all(row["stderr"] <= 60.0 for row in rows) and not notes
    line = json.loads(
        next((tmp_path / "games" / "frontline").glob("*.jsonl")).read_text().splitlines()[0]
    )
    assert line["seed"] == 7 and line["identities"]["rung_T2"] == "rung_T2"
    # a new player: its first games against the players near the middle of the field, then
    # rated like the others, the earlier games all kept
    pool.register([PoolPlayer("rung_T1.5", "rung_T1.5")])
    truth["rung_T1.5"] = -60.0
    before = sum(row["games"] for row in rows)
    rating_pool.rate(
        pool,
        "frontline",
        lambda one: player(one.name),
        _bradley_terry(truth, 32, 1),
        config,
        SCORING,
        log=logged.append,
    )
    rows, _, _ = rating_pool.table(pool, "frontline", SCORING, 1000.0)
    new = next(row for row in rows if row["player"] == "rung_T1.5")
    assert new["stderr"] <= 60.0 and sum(row["games"] for row in rows) > before


def test_a_session_holds_at_most_its_players(tmp_path):
    pool = _ladder_pool(tmp_path)
    truth = dict.fromkeys(LADDER, 0.0)
    config = tournament.TournamentConfig(
        ArenaConfig(blocks=2, block_environments=16), target_stderr=60.0, budget=64
    )
    seen = set()

    def play(field, matchups):
        seen.update(one.name for one in field)
        return _bradley_terry(truth, 16, 0)(field, matchups)

    logged = []
    rating_pool.rate(
        pool, "p", lambda one: player(one.name), play, config, SCORING, 3, logged.append
    )
    assert len(seen) <= 3 and "scripted" in seen
    assert "a session of 3 players" in logged[0]


def test_a_player_that_cannot_be_played_here_is_fitted_but_left_out_of_the_session(tmp_path):
    pool = _ladder_pool(tmp_path)
    config = tournament.TournamentConfig(
        ArenaConfig(blocks=2, block_environments=16), target_stderr=60.0, budget=32
    )
    seen = set()

    def play(field, matchups):
        seen.update(one.name for one in field)
        return _bradley_terry(dict.fromkeys(LADDER, 0.0), 16, 0)(field, matchups)

    def make(one):
        return None if one.name == "random" else player(one.name)

    rating_pool.rate(pool, "p", make, play, config, SCORING, log=lambda line: None)
    assert "random" not in seen and seen
    with pytest.raises(ValueError, match="anchors the pool"):
        rating_pool.rate(pool, "p", lambda one: None, play, config, SCORING, log=print)


# ---- importing `bnb ratings` records


def _ratings_record(path, names, identities, seed=1):
    """A `bnb ratings` record over `names` (the scripted agent first) played by the model."""
    truth = torch.linspace(0.0, 300.0, len(names), dtype=DTYPE)

    def play(matchups):
        generator = torch.Generator().manual_seed(seed)
        rows = []
        for first, second in matchups:
            won = int(
                torch.binomial(
                    torch.tensor(16.0, dtype=DTYPE),
                    win_probability(truth[first], truth[second]),
                    generator,
                )
            )
            rows.append((won, 0, 16 - won))
        return torch.tensor(rows), dict(
            kill_share=[float(row[0]) for row in rows], kills=[[1, 1]] * len(rows)
        )

    config = tournament.TournamentConfig(
        ArenaConfig(blocks=4, block_environments=16),
        target_stderr=1.0,
        budget=3 * 64,
        hopeless=math.inf,
    )
    tournament.fit_field(
        play,
        names,
        0,
        tournament.allowed_pairs(names),
        config,
        lambda line: None,
        "f",
        path,
        None,
        identities,
        None,
        SCORING,
    )


def test_imported_records_give_the_pool_their_games_once(tmp_path):
    names = [
        "scripted",
        "/box/runs/arm-a/policies/step_10/exponential_average",
        "/elsewhere/arm-a/policies/step_20/exponential_average",
        "/box/runs/arm-b/policies/step_10/exponential_average",
    ]
    identities = ["scripted", "sha256:a10", "sha256:a20", "sha256:b10"]
    record = tmp_path / "old" / "frontline.batches.jsonl"
    _ratings_record(record, names, identities)
    # a second record naming one of the same weights by another path
    other = tmp_path / "old2" / "frontline.batches.jsonl"
    _ratings_record(
        other,
        ["scripted", "/x/arm-a/policies/step_10/exponential_average"],
        ["scripted", "sha256:a10"],
        2,
    )
    pool = RatingPool.create(tmp_path / "pool")
    found = rating_pool.import_records(pool, [record, other], lambda preset: "win_loss")
    assert [entry["batches"] for entry in found.values()] == [3, 3]
    players = pool.players()[0]
    assert set(players) == {
        "scripted",
        "arm-a/step_10/exponential_average",
        "arm-a/step_20/exponential_average",
        "arm-b/step_10/exponential_average",
    }
    assert players["arm-a/step_20/exponential_average"].identity == "sha256:a20"
    again = rating_pool.import_records(pool, [record, other], lambda preset: "win_loss")
    assert [entry["batches"] for entry in again.values()] == [0, 0]
    # the pool's tally is the records' games, by pool name
    from bnb.game import VERSION

    counted = rating_pool.tally(pool.batches("frontline")[0], players, SCORING)
    reused = tournament.reuse_record(record, names, identities, scoring=SCORING)
    reused_other = tournament.reuse_record(other, names, identities, scoring=SCORING)
    assert counted.games == reused.games + reused_other.games and VERSION >= 1
    order = [
        counted.names.index(rating_pool.policy_name(name)[0]) if name != "scripted" else 0
        for name in names
    ]
    index = torch.tensor(order)
    assert torch.equal(counted.wins[index][:, index], reused.wins + reused_other.wins)


# ---- goodness of fit


def _counts_wins(scores: dict[tuple[int, int], float], count: int, games: float) -> torch.Tensor:
    wins = torch.zeros((count, count), dtype=DTYPE)
    for (first, second), score in scores.items():
        wins[first, second] = score * games
        wins[second, first] = (1.0 - score) * games
    return wins


def test_a_transitive_field_fits_and_a_cycle_is_found():
    truth = torch.tensor([0.0, 100.0, 200.0, 300.0], dtype=DTYPE)
    scores = {
        (i, j): float(win_probability(truth[i], truth[j]))
        for i in range(4)
        for j in range(i + 1, 4)
    }
    wins = _counts_wins(scores, 4, 400.0)
    fitted = goodness_of_fit.report(wins, fit_posterior(wins, 0), ["s", "a", "b", "c"])
    assert fitted["deviance"]["p_value"] > 0.5 and fitted["loops"]["p_value"] > 0.5
    # rock, paper, scissors among a, b, c (each beats the next 80%), the anchor even with all
    cycle = {(0, 1): 0.5, (0, 2): 0.5, (0, 3): 0.5, (1, 2): 0.8, (2, 3): 0.8, (1, 3): 0.2}
    wins = _counts_wins(cycle, 4, 400.0)
    fitted = goodness_of_fit.report(wins, fit_posterior(wins, 0), ["s", "a", "b", "c"])
    assert fitted["deviance"]["p_value"] < 1e-6 and fitted["loops"]["p_value"] < 1e-6
    assert {fitted["worst_pairs"][0]["first"], fitted["worst_pairs"][0]["second"]} <= {
        "a",
        "b",
        "c",
    }
    lines = goodness_of_fit.describe(fitted)
    assert lines[0].startswith("observed against predicted: deviance") and "mElo-2" in lines[1]


def test_a_tree_of_pairs_cannot_show_a_loop():
    wins = _counts_wins({(0, 1): 0.6, (1, 2): 0.7}, 3, 100.0)
    loops = goodness_of_fit.cyclic_test(wins)
    assert math.isnan(loops["p_value"]) and loops["degrees"] == 0


def test_the_pool_command_parses_its_options():
    import tyro

    from bnb.cli import Pool

    options = tyro.cli(
        Pool,
        args=[
            "rate",
            "/pool",
            "--decision",
            "a",
            "b",
            "--contrast",
            "a:b",
            "--players-per-batch",
            "4",
        ],
    )
    assert options.action == "rate" and options.pool == "/pool" and options.decision == ("a", "b")
    assert options.contrast == ("a:b",) and options.players_per_batch == 4
    added = tyro.cli(Pool, args=["add", "/pool", "p1", "p2"])
    assert added.paths == ("p1", "p2")


@pytest.mark.usefixtures("global_generator")
def test_the_pool_command_adds_rates_and_tables_scripted_players_on_the_cpu(
    tmp_path, monkeypatch, capsys
):
    """`bnb pool` end to end on a small 2v2 game (the engine eager): create, add, rate, table
    and curves."""
    import dataclasses
    import functools

    from bnb import cli
    from bnb.contracts import GameConfig
    from bnb.league import matches

    game = GameConfig(2, 2, 1500.0, episode_seconds=2.0, respawn_seconds=1.0)
    monkeypatch.setattr(cli, "rated_game", lambda preset, scale: game)
    monkeypatch.setattr(
        matches, "ArenaConfig", functools.partial(matches.ArenaConfig, length=16, compiled=False)
    )
    root = str(tmp_path / "pool")
    cli.pool(cli.Pool("create", root))
    cli.pool(cli.Pool("add", root, ("random", "rung_T2")))
    options = cli.Pool(
        "rate",
        root,
        presets=("frontline",),
        budget=16,
        blocks=2,
        block_environments=4,
        seed=3,
        device="cpu",
    )
    cli.pool(options)
    batches, _ = RatingPool(root).batches("frontline")
    assert len(batches) == 2 and all(entry["seed"] == 3 for entry in batches)
    assert all(len(entry["kill_share"]) == 2 for entry in batches)
    cli.pool(dataclasses.replace(options, action="table"))
    printed = capsys.readouterr().out
    assert "scripted" in printed and "observed against predicted" in printed
    with pytest.raises(SystemExit, match="--out"):
        cli.pool(dataclasses.replace(options, action="curves"))
