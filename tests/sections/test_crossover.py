"""The break-even count (`bnb.league.crossover`, `bnb crossover`): the fit recovers a known
crossing and gives a bound on separated outcomes, the allocation concentrates games near
the crossing, the stop rule, the per-game lineups, the engine's benching, and a small
measurement on the CPU that records every batch and resumes from its record."""

import dataclasses
import json
import math

import pytest
import torch

from bnb.contracts import GameConfig
from bnb.contracts.obs import SHIP
from bnb.game import api
from bnb.game.state import benched_of
from bnb.league import crossover as break_even
from bnb.league.matches import Arena, ArenaConfig, Player, player
from bnb.runs import checkpoint
from bnb.runtime.opponents import Rung
from tests.runs.runs_cases import tiny_model

CPU = torch.device("cpu")
# Short, small and lethal enough that most games end by the win rule (one kill per ship).
GAME = GameConfig(3, 2, 500.0, episode_seconds=20.0, respawn_seconds=1.0, kills_per_ship=1)
LAYOUT = ArenaConfig(blocks=2, block_environments=8, length=16, compiled=False)
# The same game on a short clock, for the tests that play whole batches.
SHORT = dataclasses.replace(GAME, episode_seconds=4.0)


def _scripted() -> Player:
    return Player("scripted", rung=Rung(1.0))


def _synthetic(
    true_break_even: float, slope: float, counts: list[int], games: int, seed: int
) -> dict[int, list[float]]:
    """Tallies of games drawn from a logistic curve in log(count) crossing one half at
    `true_break_even`: each game's expected score the curve's, some of it in draws."""
    generator = torch.Generator().manual_seed(seed)
    blocks = []
    for count in counts:
        p = 1 / (1 + math.exp(-slope * math.log(count / true_break_even)))
        draw_chance = 0.4 * min(p, 1 - p)
        win_chance = p - draw_chance / 2
        uniform = torch.rand(games, generator=generator)
        wins = int((uniform < win_chance).sum())
        draws = int(((uniform >= win_chance) & (uniform < win_chance + draw_chance)).sum())
        blocks.append(dict(count=count, wins=wins, draws=draws, losses=games - wins - draws))
    return break_even.tallies(blocks)


# ---- the fit


def test_the_fit_recovers_a_known_crossing_within_its_standard_error():
    misses = 0
    for seed in range(20):
        tallies = _synthetic(8.0, 5.0, [4, 6, 8, 10, 12, 16], 200, seed)
        found = break_even.fit(tallies)
        assert found.status == "crossing" and found.slope > 0
        assert 0.05 < found.stderr < 1.0
        misses += abs(found.break_even - 8.0) > 2 * found.stderr
    assert misses <= 3  # about 5% expected outside two standard errors


def test_the_fit_crosses_where_the_curve_is_one_half_and_the_ratio_follows():
    # exact expected scores, no noise: the fit is the curve itself
    blocks = []
    for count in (5, 8, 13):
        p = 1 / (1 + math.exp(-4.0 * math.log(count / 8.0)))
        blocks.append(dict(count=count, wins=1000 * p, draws=0, losses=1000 * (1 - p)))
    found = break_even.fit(break_even.tallies(blocks))
    assert found.break_even == pytest.approx(8.0, rel=1e-6)
    assert found.slope == pytest.approx(4.0, rel=1e-6)
    ratio, ratio_stderr = found.ratio(20)
    assert ratio == pytest.approx(2.5, rel=1e-6)
    assert ratio_stderr == pytest.approx(ratio * found.stderr / found.break_even)


def test_a_draw_counts_half_a_win():
    drawn = break_even.tallies([dict(count=4, wins=0, draws=10, losses=0)])
    assert drawn == {4: [5.0, 10.0]}


def test_separated_outcomes_give_a_bound():
    def tallies(*rows):
        return break_even.tallies(
            [
                dict(count=count, wins=won, draws=drawn, losses=lost)
                for count, won, drawn, lost in rows
            ]
        )

    every_win = break_even.fit(tallies((6, 10, 0, 0), (10, 10, 0, 0)))
    assert (every_win.status, every_win.lower, every_win.upper) == ("separated", None, 6)
    every_loss = break_even.fit(tallies((6, 0, 0, 10), (20, 0, 0, 10)))
    assert (every_loss.lower, every_loss.upper) == (20, None)
    between = break_even.fit(tallies((6, 0, 0, 10), (10, 0, 0, 10), (14, 10, 0, 0)))
    assert (between.status, between.lower, between.upper) == ("separated", 10, 14)
    assert between.ratio(20) == (None, None)
    overlapping = break_even.fit(tallies((6, 1, 0, 9), (10, 3, 0, 7), (14, 10, 0, 0)))
    assert overlapping.status == "crossing"
    assert break_even.fit({}).status == "empty"
    assert break_even.fit(tallies((6, 10, 0, 0), (14, 0, 0, 10))).status == "falling"


# ---- the allocation and the stop rule


CONFIG = break_even.CrossoverConfig()


def test_the_allocation_concentrates_blocks_at_the_crossing():
    tallies = _synthetic(8.0, 5.0, [6, 10, 14, 20], 256, seed=1)
    found = break_even.fit(tallies)
    counts = break_even.allocate(found, tallies, CONFIG, 8, 128)
    assert len(counts) == 8
    assert all(abs(math.log(count / found.break_even)) < 0.3 for count in counts)


def test_an_adaptive_measurement_puts_most_games_near_the_crossing():
    generator = torch.Generator().manual_seed(3)
    truth, slope, block_games = 9.3, 5.0, 64
    blocks: list[dict] = []
    counts = break_even._spread(CONFIG.bracket_counts(), 8)
    for _ in range(8):
        for count in counts:
            p = 1 / (1 + math.exp(-slope * math.log(count / truth)))
            wins = int((torch.rand(block_games, generator=generator) < p).sum())
            blocks.append(dict(count=count, wins=wins, draws=0, losses=block_games - wins))
        tallies = break_even.tallies(blocks)
        found = break_even.fit(tallies)
        counts = break_even.allocate(found, tallies, CONFIG, 8, block_games)
    near = sum(block_games for block in blocks if 7 <= block["count"] <= 12)
    assert near / (len(blocks) * block_games) > 0.7
    assert abs(found.break_even - truth) < 3 * found.stderr
    assert found.stderr < 0.5


def test_separated_outcomes_steer_into_or_beyond_the_bound():
    wins = break_even.Fit("separated", lower=None, upper=6)
    assert break_even.allocate(wins, {6: [10, 10]}, CONFIG, 4, 8) == [3, 3, 3, 3]
    losses = break_even.Fit("separated", lower=6, upper=None)
    assert break_even.allocate(losses, {6: [0, 10]}, CONFIG, 2, 8) == [12, 12]
    inside = break_even.Fit("separated", lower=6, upper=14)
    assert set(break_even.allocate(inside, {}, CONFIG, 6, 8)) == {8, 9, 10}
    adjacent = break_even.Fit("separated", lower=9, upper=10)
    assert break_even.allocate(adjacent, {}, CONFIG, 4, 8) == [9, 10, 9, 10]
    assert break_even.allocate(break_even.Fit("empty"), {}, CONFIG, 8, 8) == [6, 10, 14, 20] * 2


def test_the_stop_rule():
    assert break_even.stop_reason(break_even.Fit("crossing", 8.0, 0.25), 1024, CONFIG) == "target"
    assert break_even.stop_reason(break_even.Fit("crossing", 8.0, 0.5), 1024, CONFIG) is None
    assert break_even.stop_reason(break_even.Fit("crossing", 8.0, 0.5), 8192, CONFIG) == "budget"
    floor = break_even.Fit("separated", lower=None, upper=1)
    ceiling = break_even.Fit("separated", lower=20, upper=None)
    assert break_even.stop_reason(floor, 64, CONFIG) == "bound"
    assert break_even.stop_reason(ceiling, 64, CONFIG) == "bound"
    open_bound = break_even.Fit("separated", lower=None, upper=6)
    assert break_even.stop_reason(open_bound, 64, CONFIG) is None


# ---- the lineups and the engine


def test_each_game_draws_its_lineup_from_the_types_keyed_by_seed_and_batch():
    one = break_even.lineup_draw(0, 3, 256, 20, (0, 1))
    assert one.shape == (256, 20) and one.dtype == torch.int32
    assert set(one.unique().tolist()) == {0, 1}
    assert 0.45 < one.float().mean() < 0.55
    assert torch.equal(one, break_even.lineup_draw(0, 3, 256, 20, (0, 1)))
    assert not torch.equal(one, break_even.lineup_draw(0, 4, 256, 20, (0, 1)))
    assert not torch.equal(one, break_even.lineup_draw(1, 3, 256, 20, (0, 1)))


@pytest.mark.usefixtures("global_generator")
def test_an_engine_with_game_lineups_flies_each_games_lineup_on_both_teams():
    arena = Arena([_scripted(), _scripted()], SHORT, CPU, LAYOUT, benching=True, game_lineups=True)
    lineups = break_even.lineup_draw(0, 0, LAYOUT.games, SHORT.ships_per_team, (0, 1))
    arena.play([(0, 1)] * LAYOUT.blocks, [(3, 3), (2, 3)], lineups)
    assert torch.equal(arena.rollout.ship_type.cpu(), lineups.repeat(1, 2))
    with pytest.raises(ValueError, match="per-game lineups"):
        arena.play([(0, 1)] * LAYOUT.blocks, [(3, 3), (2, 3)])
    with pytest.raises(ValueError, match="per-game lineups"):
        Arena(
            [_scripted(), Player("typed", rung=Rung(1.0), ship_type=0)],
            GAME,
            CPU,
            LAYOUT,
            game_lineups=True,
        )


@pytest.mark.usefixtures("global_generator")
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


@pytest.mark.usefixtures("global_generator")
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


@pytest.mark.usefixtures("global_generator")
def test_an_engine_without_benching_refuses_counts():
    arena = Arena([_scripted(), _scripted()], GAME, CPU, LAYOUT)
    with pytest.raises(ValueError, match="benching"):
        arena.play([(0, 1), (0, 1)], [(3, 3), (3, 3)])


@pytest.mark.usefixtures("global_generator")
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


# ---- a measurement


SMALL = break_even.CrossoverConfig(
    opponent_count=3, minimum_count=1, maximum_count=3, bracket=(1, 3), budget_games=32
)


@pytest.mark.usefixtures("global_generator")
def test_a_measurement_records_every_batch_and_resumes_from_its_record(tmp_path):
    record = tmp_path / "small.batches.jsonl"
    printed: list[str] = []
    first = break_even.measure(
        _scripted(), _scripted(), SHORT, CPU, LAYOUT, SMALL, record, printed.append
    )
    lines = [json.loads(line) for line in record.read_text().splitlines()]
    assert first["stop"] in ("budget", "target", "bound")
    assert len(lines) == first["batches"] and first["games"] == 16 * len(lines)
    assert [line["batch"] for line in lines] == list(range(len(lines)))
    assert lines[0]["blocks"][0]["count"] == 1 and lines[0]["blocks"][1]["count"] == 3
    assert all(
        block["wins"] + block["draws"] + block["losses"] == LAYOUT.block_environments
        for line in lines
        for block in line["blocks"]
    )
    assert sum(row["games"] for row in first["counts"]) == first["games"]
    if first["stop"] == "budget":
        assert any("ETA" in text for text in printed[:-1]) or len(printed) == 1
    # a rerun with a larger budget reuses the record's games and continues past them
    larger = dataclasses.replace(SMALL, budget_games=64, target_stderr=1e-6)
    printed.clear()
    second = break_even.measure(
        _scripted(), _scripted(), SHORT, CPU, LAYOUT, larger, record, printed.append
    )
    resumed = [json.loads(line) for line in record.read_text().splitlines()]
    assert resumed[: len(lines)] == lines
    assert printed[0].startswith(f"reusing {len(lines)} batches")
    assert [line["batch"] for line in resumed] == list(range(len(resumed)))
    assert second["games"] >= 64 and second["batches"] == len(resumed)
    # a rerun of a finished measurement plays nothing
    third = break_even.measure(
        _scripted(), _scripted(), SHORT, CPU, LAYOUT, larger, record, printed.append
    )
    assert third["games"] == second["games"]
    assert len(record.read_text().splitlines()) == len(resumed)
    # another measurement in the same record starts its own batches
    other = dataclasses.replace(LAYOUT, seed=1)
    fourth = break_even.measure(
        _scripted(), _scripted(), SHORT, CPU, other, SMALL, record, printed.append
    )
    assert fourth["batches"] >= 1 and fourth["games"] == 16 * fourth["batches"]


@pytest.mark.usefixtures("global_generator")
def test_a_network_plays_a_measurement(tmp_path):
    game = GameConfig(2, 2, 700.0, episode_seconds=4.0, respawn_seconds=1.0)
    directory = tmp_path / "policy"
    checkpoint.save_policy(directory, tiny_model("yemong", seed=1, game=game))
    layout = dataclasses.replace(LAYOUT, block_environments=4)
    config = break_even.CrossoverConfig(
        opponent_count=2, minimum_count=1, maximum_count=2, bracket=(1, 2), budget_games=8
    )
    result = break_even.measure(
        player(str(directory)),
        _scripted(),
        game,
        CPU,
        layout,
        config,
        tmp_path / "record.jsonl",
        lambda text: None,
    )
    assert result["policy_identity"].startswith("sha256:")
    assert result["opponent_identity"] == "scripted" and result["games"] == 8


def test_the_game_must_be_the_larger_fleets_size():
    with pytest.raises(ValueError, match="ships per team"):
        break_even.measure(
            _scripted(), _scripted(), GAME, CPU, LAYOUT, CONFIG, None, lambda text: None
        )


# ---- the command


def test_the_command_parses_its_options_with_the_decided_defaults():
    import tyro

    from bnb.cli import COMMANDS, Crossover

    options = tyro.cli(Crossover, args=["policy"])
    assert options.policy == "policy" and options.opponent == "scripted"
    assert (options.opponent_count, options.minimum_count, options.maximum_count) == (20, 1, 20)
    assert options.presets == (
        "real_scale_frontline",
        "real_scale_deathmatch",
        "real_scale_elimination",
    )
    assert options.lineup_types == ("Yongma", "Jujak") and options.target_stderr == 0.3
    assert "crossover" in COMMANDS


@pytest.mark.usefixtures("global_generator")
def test_the_command_measures_each_preset_and_records_the_result(tmp_path, monkeypatch, capsys):
    import functools

    from bnb import cli
    from bnb.league import matches

    monkeypatch.setitem(api.PRESETS, "tiny", SHORT)
    monkeypatch.setattr(
        matches, "ArenaConfig", functools.partial(ArenaConfig, length=16, compiled=False)
    )
    names = api.type_names(SHORT.ship_set)
    options = cli.Crossover(
        policy="scripted",
        opponent_count=3,
        minimum_count=1,
        maximum_count=3,
        presets=("tiny",),
        bracket=(1, 3),
        budget_games=16,
        lineup_types=names[:2],
        blocks=2,
        block_environments=8,
        device="cpu",
        out=str(tmp_path),
    )
    cli.crossover(options)
    printed = capsys.readouterr().out
    assert "tiny (1 of 1): scripted with 1-3 ships against 3 scripted, 3v3 game" in printed
    record = tmp_path / "artifacts" / "crossover" / "tiny.batches.jsonl"
    assert len(record.read_text().splitlines()) == 1
    (directory,) = [
        path for path in (tmp_path / "artifacts" / "crossover").iterdir() if path.is_dir()
    ]
    result = json.loads((directory / "result.json").read_text())
    assert result["payload"]["presets"]["tiny"]["games"] == 16
    assert result["payload"]["lineup_types"] == list(names[:2])
    assert "command" in result["provenance"]
