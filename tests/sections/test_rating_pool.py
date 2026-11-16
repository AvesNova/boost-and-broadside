"""The shared rating pool (league spec 6, `bnb.league.rating_pool`): stable names, one file
per writer merged on read, the import of `bnb ratings` records, a rating session over the
pool, a new checkpoint's placement, and the goodness-of-fit report
(`bnb.league.goodness_of_fit`), on the CPU with scripted players and a Bradley-Terry model
of the games."""

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
    rows, check, notes = rating_pool.table(pool, "frontline", SCORING)
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
    rows, _, _ = rating_pool.table(pool, "frontline", SCORING)
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


def test_a_contrast_session_plays_named_policies_against_the_incumbent(tmp_path, monkeypatch):
    """The averaging sweep's pool (box/eval_queue.sh): averages registered under their own
    names (`bnb pool add PATH --name`, no run's step directory) are played by `bnb pool
    rate`, and a session targeting each candidate's difference from the incumbent plays
    them until every contrast's standard error is met or the budget is spent."""
    from bnb import cli

    averages = ["incumbent", "uniform", "window", "halflife"]
    truth = {"scripted": 0.0, "r/step_100/live": 380.0}
    truth |= {f"r/average/{name}/step_100": 400.0 + 15 * k for k, name in enumerate(averages)}
    identities = {}
    pool = RatingPool.create(tmp_path / "pool")
    for name in truth:
        if name == "scripted":
            continue
        directory = tmp_path / "runs" / name.replace("/step_100/live", "/policies/step_100/live")
        directory.mkdir(parents=True)
        identities[str(directory)] = f"sha256:{len(identities)}"
    monkeypatch.setattr(rating_pool, "policy_identity", lambda path: identities[str(path)])
    monkeypatch.setattr(  # one network build, read from no files
        rating_pool,
        "network_groups",
        lambda field: [None if one.policy is None else 0 for one in field],
    )
    for path in identities:
        named = None if "/policies/" in path else path.split("/runs/")[1]
        pool.add_policy(path, named)
    players, _ = pool.players()
    assert players["r/average/incumbent/step_100"].run is None  # as on the box
    incumbent = "r/average/incumbent/step_100"
    contrasts = tuple(
        (f"r/average/{name}/step_100", incumbent) for name in averages if name != "incumbent"
    )
    budget = 128 * 120
    config = tournament.TournamentConfig(
        ArenaConfig(blocks=4, block_environments=32),
        target_stderr=45.0,
        budget=budget,
        contrasts=contrasts,
        contrast_stderr=20.0,
    )
    played = []

    def play(field, matchups):
        played.extend({field[a].name, field[b].name} for a, b in matchups)
        return _bradley_terry(truth, 32, 2)(field, matchups)

    logged = []
    rating_pool.rate(
        pool,
        "frontline",
        lambda one: cli._pool_player(one, tmp_path / "runs", (), "frontline"),
        play,
        config,
        SCORING,
        log=logged.append,
    )
    assert f"a session of {len(truth)} players" in logged[0], logged[0]
    assert not any("left out of the session" in line for line in logged)
    for first, second in contrasts:
        assert {first, second} in played or any(
            first in pair and second not in pair for pair in played
        )
    counted, posterior, _, _ = rating_pool.fit(pool, "frontline", SCORING)
    targets = config.targets(counted.names)
    errors, goals = targets.standard_errors(posterior)
    contrast_errors = errors[-len(contrasts) :]
    assert torch.isfinite(contrast_errors).all()
    games = 32 * len(played)
    assert bool((contrast_errors <= 20.0).all()) or games + 128 > budget, contrast_errors
    assert "every target of the session is met" in logged[-1], logged[-3:]


# ---- importing `bnb ratings` records


def _ratings_record(path, names, identities, seed=1):
    """A `bnb ratings` record over `names` (Imugi first) played by the model."""
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


def _placed(rated: dict[str, float], new: list[str]) -> dict[str, float]:
    """The placement guesses of the `new` players (pool names, no games) in a pool whose
    `rated` players (pool names) each played Imugi to the score their rating
    gives, so each is fitted at its rating exactly."""
    names = ["scripted", *rated, *new]
    count = len(names)
    wins = torch.zeros((count, count), dtype=DTYPE)
    for index, rating in enumerate(rated.values(), start=1):
        share = float(win_probability(torch.tensor(rating, dtype=DTYPE), torch.tensor(0.0)))
        wins[index, 0], wins[0, index] = 1000 * share, 1000 * (1 - share)
    posterior = fit_posterior(wins, 0)
    players = {"scripted": PoolPlayer("scripted", "scripted")}
    for name in names[1:]:
        _, run, step, kind = rating_pool.policy_name(
            f"/x/{name.replace('/step_', '/policies/step_')}"
        )
        players[name] = PoolPlayer(name, name, run, step, kind)
    guesses = rating_pool.placement(posterior, names, players)
    return {name: float(guesses[names.index(name)]) for name in new}


def test_a_new_checkpoint_is_placed_by_the_other_runs_at_its_steps():
    """Placement (allocation only): the mean of the other runs' checkpoints at the same
    steps; between steps, interpolated; past the last, the nearest steps'; with no other
    run, the run's previous checkpoint; else the anchor. Another weights kind is not used."""
    average = "exponential_average"
    rated = {
        f"a/step_100/{average}": 100.0,
        f"b/step_100/{average}": 300.0,
        f"a/step_200/{average}": 400.0,
        f"b/step_200/{average}": 600.0,
        "b/step_100/live": 900.0,
    }
    new = [f"c/step_{step}/{average}" for step in (100, 150, 300, 50)]
    placed = _placed(rated, new)
    assert placed[new[0]] == pytest.approx(200.0, abs=1e-6)  # the mean at 100
    assert placed[new[1]] == pytest.approx(350.0, abs=1e-6)  # halfway from 200 to 500
    assert placed[new[2]] == pytest.approx(500.0, abs=1e-6)  # the nearest, 200's mean
    assert placed[new[3]] == pytest.approx(200.0, abs=1e-6)  # the nearest, 100's mean
    # one run alone: its previous checkpoint; none earlier: the anchor
    alone = {f"a/step_100/{average}": 100.0, f"a/step_200/{average}": 400.0}
    placed = _placed(alone, [f"a/step_300/{average}", f"a/step_50/{average}"])
    assert placed == {f"a/step_300/{average}": pytest.approx(400.0), f"a/step_50/{average}": 0.0}


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


def test_a_pool_may_draw_each_games_lineup_and_an_older_pool_draws_none(tmp_path):
    drawn = RatingPool.create(tmp_path / "drawn", lineup_types=("Yongma", "Jujak"))
    assert drawn.settings["lineup_types"] == ["Yongma", "Jujak"]
    with pytest.raises(ValueError, match="not both"):
        RatingPool.create(tmp_path / "both", ("Yongma",), lineup_types=("Yongma", "Jujak"))
    older = tmp_path / "older"
    RatingPool.create(older)
    record = older / rating_pool.POOL_RECORD
    settings = json.loads(record.read_text())
    del settings["lineup_types"]  # written before drawn lineups existed
    record.write_text(json.dumps(settings))
    assert RatingPool(older).settings["lineup_types"] == []
    RatingPool.create(older)  # the same conditions: opened
    with pytest.raises(ValueError, match="other conditions"):
        RatingPool.create(older, lineup_types=("Yongma",))


def test_a_runs_snapshots_are_its_files_at_the_schedule(tmp_path):
    from bnb.runtime.snapshots import SnapshotSchedule

    run = tmp_path / "chain-rl"
    update = 300
    for step in range(0, 3001, update):  # every update's files, from the starting weights
        for kind in ("live", "exponential_average"):
            (run / "policies" / f"step_{step}" / kind).mkdir(parents=True)
    schedule = SnapshotSchedule((500, 1000), 1000)
    chosen = rating_pool.scheduled_policies(run, ("exponential_average",), schedule)
    # 500 -> 600, 1000 -> 1200, 2000 -> 2100, 3000; and the last
    assert [path.parent.name for path in chosen] == [
        "step_600",
        "step_1200",
        "step_2100",
        "step_3000",
    ]
    both = rating_pool.scheduled_policies(run, ("exponential_average", "live"), schedule)
    assert [path.name for path in both[:2]] == ["exponential_average", "live"]
    with pytest.raises(ValueError, match="no policy files"):
        rating_pool.scheduled_policies(tmp_path / "empty", ("live",), schedule)


def test_a_head_to_head_condition_is_a_rated_game_and_break_even_is_not():
    from bnb import cli
    from bnb.game.api import PRESETS

    base = PRESETS["real_scale_deathmatch"]
    game = cli.named_game("deathmatch_10v10")
    assert game.ships_per_team == 10 and game.arena_radius == base.arena_radius
    assert game.mode == base.mode
    assert cli.named_game("frontline_5v5") == PRESETS["real_scale_frontline"]
    assert cli.named_game("real_scale_elimination") is PRESETS["real_scale_elimination"]
    with pytest.raises(SystemExit, match="bnb crossover"):
        cli.named_game("deathmatch_break_even")


@pytest.mark.usefixtures("global_generator")
def test_the_pool_rates_drawn_lineups_and_draws_its_curves_on_the_cpu(tmp_path, monkeypatch):
    """`bnb pool` with each game's lineup drawn (a small 2v2 game, the engine eager), a run's
    scheduled snapshots added by `--runs`, and the curves drawn as small multiples."""
    import csv
    import dataclasses
    import functools

    from bnb import cli
    from bnb.contracts import GameConfig
    from bnb.figures import ladder
    from bnb.league import matches

    game = GameConfig(2, 2, 1500.0, episode_seconds=2.0, respawn_seconds=1.0)
    monkeypatch.setattr(cli, "rated_game", lambda preset, scale: game)
    monkeypatch.setattr(
        matches, "ArenaConfig", functools.partial(matches.ArenaConfig, length=16, compiled=False)
    )
    played = []
    original = matches.Arena.play

    def play(self, matchups, active=None, lineups=None):
        played.append(lineups)
        return original(self, matchups, active, lineups)

    monkeypatch.setattr(matches.Arena, "play", play)
    root = str(tmp_path / "pool")
    cli.pool(cli.Pool("create", root, lineup_types=("Fighter", "Heavy")))
    cli.pool(cli.Pool("add", root, ("random", "rung_T2")))
    with pytest.raises(SystemExit, match="no policy files"):
        cli.pool(cli.Pool("add", root, runs=(str(tmp_path / "missing-run"),)))
    options = cli.Pool(
        "rate",
        root,
        presets=("deathmatch_2v2",),
        budget=16,
        blocks=2,
        block_environments=4,
        seed=3,
        device="cpu",
    )
    cli.pool(options)
    assert len(played) == 2 and all(lineups is not None for lineups in played)
    assert tuple(played[0].shape) == (8, 2) and not torch.equal(played[0], played[1])
    assert len(RatingPool(root).batches("deathmatch_2v2")[0]) == 2
    out = tmp_path / "curves.csv"
    figure = tmp_path / "curves.png"
    cli.pool(dataclasses.replace(options, action="curves", out=str(out), figure=str(figure)))
    assert figure.is_file() and list(csv.DictReader(out.open())) == []  # no policies

    rows = [
        dict(
            preset="deathmatch_2v2",
            run=run,
            kind="exponential_average",
            step=step,
            calibrated_elo=elo,
            stderr=30.0,
            status="rated",
        )
        for run, step, elo in (("r-bc", 10, 900.0), ("r-bc", 20, 950.0), ("r-rl", 5, 1100.0))
    ]
    assert ladder.chain_offsets(rows) == {"r-bc": 0, "r-rl": 20}
    result = tmp_path / "break-even" / "r-rl" / "step_5" / "artifacts" / "crossover" / "t1"
    result.mkdir(parents=True)
    (result / "result.json").write_text(
        json.dumps(
            dict(
                payload=dict(presets=dict(real_scale_deathmatch=dict(ratio=1.3, ratio_stderr=0.1)))
            )
        )
    )
    points = ladder.break_even_points(tmp_path / "break-even")
    assert points == [dict(run="r-rl", step=5, mode="deathmatch", ratio=1.3, stderr=0.1)]
    drawn = ladder.draw(rows, ("deathmatch_2v2", "deathmatch_10v10"), tmp_path / "f.png", points)
    assert drawn.is_file()
