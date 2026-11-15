"""The break-even lineup (`bnb.league.crossover`, `bnb crossover`): the lineups on the log
ratio axis, the fit recovers a known crossing on either side of parity and gives a bound on
separated outcomes, the allocation starts at parity, picks the shrinking side from the games
and concentrates the games near the crossing, the stop rule, the per-game lineups, the
engine's benching, and a small measurement on the CPU that records every batch and resumes
from its record."""

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


def _curve(true_ratio: float, slope: float, lineup) -> float:
    """The policy's expected score at a lineup on a logistic curve in the log ratio crossing
    one half at `true_ratio`."""
    return 1 / (1 + math.exp(-slope * (break_even.log_ratio(lineup) - math.log(true_ratio))))


def _blocks(true_ratio: float, slope: float, lineups, games: int, generator) -> list[dict]:
    """A block of `games` games per lineup drawn from `_curve`, some of the score in draws."""
    blocks = []
    for lineup in lineups:
        p = _curve(true_ratio, slope, lineup)
        draw_chance = 0.4 * min(p, 1 - p)
        win_chance = p - draw_chance / 2
        uniform = torch.rand(games, generator=generator)
        wins = int((uniform < win_chance).sum())
        draws = int(((uniform >= win_chance) & (uniform < win_chance + draw_chance)).sum())
        blocks.append(
            dict(
                policy_count=lineup[0],
                opponent_count=lineup[1],
                wins=wins,
                draws=draws,
                losses=games - wins - draws,
            )
        )
    return blocks


def _synthetic(true_ratio: float, slope: float, lineups, games: int, seed: int) -> dict:
    """Tallies of `_blocks`."""
    generator = torch.Generator().manual_seed(seed)
    return break_even.tallies(_blocks(true_ratio, slope, lineups, games, generator))


def _rows(*rows) -> dict:
    """Tallies of (policy's ships, opponent's ships, wins, draws, losses) rows."""
    return break_even.tallies(
        [
            dict(policy_count=mine, opponent_count=theirs, wins=won, draws=drawn, losses=lost)
            for mine, theirs, won, drawn, lost in rows
        ]
    )


# ---- the lineups


def test_the_lineups_run_along_the_ratio_with_the_weaker_side_fixed():
    config = break_even.CrossoverConfig(minimum_count=3)
    lineups = config.lineups()
    ratios = [break_even.log_ratio(lineup) for lineup in lineups]
    assert ratios == sorted(ratios) and len(set(ratios)) == len(lineups) == 2 * 17 + 1
    assert lineups[0] == (3, 20) and lineups[-1] == (20, 3) and (20, 20) in lineups
    assert all(20 in lineup and min(lineup) >= 3 and max(lineup) <= 20 for lineup in lineups)
    assert config.ships_per_team == 20 and config.parity == (20, 20)
    clipped = break_even.CrossoverConfig(minimum_count=3, bracket=(1, 10, 30))
    assert clipped.bracket_lineups("policy") == [(3, 20), (10, 20), (20, 20)]
    assert clipped.bracket_lineups("opponent") == [(20, 20), (20, 10), (20, 3)]
    assert len(clipped.bracket_lineups("both")) == 5
    with pytest.raises(ValueError, match="fixed count"):
        break_even.CrossoverConfig(minimum_count=21)


# ---- the fit


@pytest.mark.parametrize("true_ratio", [0.4, 1.6])
def test_the_fit_recovers_a_known_crossing_within_its_standard_error(true_ratio):
    """A stronger policy's (8 ships hold 20) and a weaker one's (20 hold 12.5)."""
    lineups = (
        [(4, 20), (6, 20), (8, 20), (10, 20), (12, 20), (16, 20)]
        if true_ratio < 1
        else [(20, 20), (20, 16), (20, 13), (20, 11), (20, 9), (20, 7)]
    )
    misses = 0
    for seed in range(20):
        found = break_even.fit(_synthetic(true_ratio, 5.0, lineups, 200, seed))
        assert found.status == "crossing" and found.slope > 0
        assert 0.005 < found.stderr < 0.15
        misses += abs(found.break_even - math.log(true_ratio)) > 2 * found.stderr
    assert misses <= 3  # about 5% expected outside two standard errors


def test_the_fit_crosses_where_the_curve_is_one_half_on_either_side_of_parity():
    # exact expected scores, no noise, both regimes in one fit: the fit is the curve itself
    blocks = []
    for lineup in ((10, 20), (20, 20), (20, 13)):
        p = _curve(1.2, 4.0, lineup)
        blocks.append(
            dict(
                policy_count=lineup[0],
                opponent_count=lineup[1],
                wins=1000 * p,
                draws=0,
                losses=1000 * (1 - p),
            )
        )
    found = break_even.fit(break_even.tallies(blocks))
    assert found.break_even == pytest.approx(math.log(1.2), rel=1e-6)
    assert found.slope == pytest.approx(4.0, rel=1e-6)
    ratio, ratio_stderr = found.ratio()
    assert ratio == pytest.approx(1.2, rel=1e-6)
    assert ratio_stderr == pytest.approx(ratio * found.stderr)
    assert found.lineup(20) == pytest.approx((20.0, 20 / 1.2))
    stronger = dataclasses.replace(found, break_even=math.log(0.45))
    assert stronger.lineup(20) == pytest.approx((9.0, 20.0))


def test_a_draw_counts_half_a_win():
    assert _rows((4, 20, 0, 10, 0)) == {(4, 20): [5.0, 10.0]}


def test_separated_outcomes_give_a_bound():
    every_win = break_even.fit(_rows((6, 20, 10, 0, 0), (10, 20, 10, 0, 0)))
    assert (every_win.status, every_win.lower, every_win.upper) == ("separated", None, (6, 20))
    every_loss = break_even.fit(_rows((6, 20, 0, 0, 10), (20, 20, 0, 0, 10)))
    assert (every_loss.lower, every_loss.upper) == ((20, 20), None)
    across = break_even.fit(_rows((20, 20, 0, 0, 10), (20, 12, 10, 0, 0)))
    assert (across.lower, across.upper) == ((20, 20), (20, 12))
    between = break_even.fit(_rows((6, 20, 0, 0, 10), (10, 20, 0, 0, 10), (14, 20, 10, 0, 0)))
    assert (between.status, between.lower, between.upper) == ("separated", (10, 20), (14, 20))
    assert between.ratio() == (None, None) and between.lineup(20) is None
    overlapping = break_even.fit(_rows((6, 20, 1, 0, 9), (10, 20, 3, 0, 7), (14, 20, 10, 0, 0)))
    assert overlapping.status == "crossing"
    assert break_even.fit({}).status == "empty"
    assert break_even.fit(_rows((6, 20, 10, 0, 0), (20, 14, 0, 0, 10))).status == "falling"
    parity_only = break_even.fit(_rows((20, 20, 6, 0, 4)))
    assert (parity_only.status, parity_only.lower, parity_only.upper) == ("separated", None, None)


# ---- the allocation and the stop rule


CONFIG = break_even.CrossoverConfig()


def test_the_first_batch_plays_parity_and_the_games_pick_the_shrinking_side():
    assert break_even.allocate(break_even.Fit("empty"), {}, CONFIG, 4, 8) == [(20, 20)] * 4
    winning = _rows((20, 20, 6, 0, 4))
    found = break_even.fit(winning)
    assert break_even.allocate(found, winning, CONFIG, 4, 8) == [
        (6, 20),
        (10, 20),
        (14, 20),
        (20, 20),
    ]
    losing = _rows((20, 20, 3, 2, 5))
    assert break_even.allocate(break_even.fit(losing), losing, CONFIG, 4, 8) == [
        (20, 20),
        (20, 14),
        (20, 10),
        (20, 6),
    ]
    even = _rows((20, 20, 4, 2, 4))
    lineups = break_even.allocate(break_even.fit(even), even, CONFIG, 7, 8)
    assert sorted(lineups, key=break_even.log_ratio) == CONFIG.bracket_lineups("both")


def test_the_allocation_concentrates_blocks_at_the_crossing():
    tallies = _synthetic(0.4, 5.0, [(6, 20), (10, 20), (14, 20), (20, 20)], 256, seed=1)
    found = break_even.fit(tallies)
    lineups = break_even.allocate(found, tallies, CONFIG, 8, 128)
    assert len(lineups) == 8
    assert all(abs(break_even.log_ratio(one) - found.break_even) < 0.3 for one in lineups)


@pytest.mark.parametrize(
    ("true_ratio", "minimum_count"), [(9.3 / 20, 1), (20 / 13, 3), (20 / 19.5, 3), (0.2, 3)]
)
def test_a_simulated_measurement_finds_the_ratio_on_either_side_of_parity(
    true_ratio, minimum_count
):
    """Synthetic games from a known logistic in the log ratio, a stronger policy's, a weaker
    one's and one near parity, played batch by batch through the allocation from the first
    (parity) batch: the fit finds the ratio within its standard error, most games go near
    the crossing, and no lineup has more than 20 a side or fewer than the floor."""
    config = dataclasses.replace(CONFIG, minimum_count=minimum_count)
    generator = torch.Generator().manual_seed(3)
    blocks: list[dict] = []
    found, tallies = break_even.Fit("empty"), {}
    for _ in range(12):
        lineups = break_even.allocate(found, tallies, config, 8, 64)
        blocks += _blocks(true_ratio, 5.0, lineups, 64, generator)
        tallies = break_even.tallies(blocks)
        found = break_even.fit(tallies)
    assert blocks[0]["policy_count"] == blocks[0]["opponent_count"] == 20
    assert found.status == "crossing"
    assert abs(found.break_even - math.log(true_ratio)) < 3 * found.stderr
    assert found.stderr < 0.05
    played = {(one["policy_count"], one["opponent_count"]) for one in blocks}
    assert all(max(one) <= 20 and min(one) >= minimum_count for one in played)
    assert all(20 in one for one in played)
    if true_ratio > 1.1:
        assert all(one[0] == 20 for one in played)  # the policy's ships never shrink
    if true_ratio < 0.9:
        assert all(one[1] == 20 for one in played)  # nor the opponent's
    near = sum(
        64
        for one in blocks
        if abs(
            break_even.log_ratio((one["policy_count"], one["opponent_count"])) - found.break_even
        )
        < 0.3
    )
    assert near / (64 * len(blocks)) > 0.6


def test_separated_outcomes_steer_into_or_beyond_the_bound():
    wins = break_even.Fit("separated", lower=None, upper=(6, 20))
    assert break_even.allocate(wins, {}, CONFIG, 4, 8) == [(3, 20)] * 4
    losses = break_even.Fit("separated", lower=(6, 20), upper=None)
    assert break_even.allocate(losses, {}, CONFIG, 2, 8) == [(12, 20)] * 2
    lost_at_parity = break_even.Fit("separated", lower=(20, 20), upper=None)
    assert break_even.allocate(lost_at_parity, {}, CONFIG, 2, 8) == [(20, 10)] * 2
    won_at_parity = break_even.Fit("separated", lower=None, upper=(20, 20))
    assert break_even.allocate(won_at_parity, {}, CONFIG, 2, 8) == [(10, 20)] * 2
    inside = break_even.Fit("separated", lower=(6, 20), upper=(14, 20))
    assert set(break_even.allocate(inside, {}, CONFIG, 6, 8)) == {(8, 20), (9, 20), (10, 20)}
    across = break_even.Fit("separated", lower=(18, 20), upper=(20, 18))
    assert set(break_even.allocate(across, {}, CONFIG, 3, 8)) == {(19, 20), (20, 20), (20, 19)}
    adjacent = break_even.Fit("separated", lower=(9, 20), upper=(10, 20))
    assert break_even.allocate(adjacent, {}, CONFIG, 4, 8) == [(9, 20), (10, 20)] * 2
    floored = dataclasses.replace(CONFIG, minimum_count=3)
    near_floor = break_even.Fit("separated", lower=None, upper=(4, 20))
    assert break_even.allocate(near_floor, {}, floored, 2, 8) == [(3, 20)] * 2


def test_the_stop_rule():
    assert break_even.stop_reason(break_even.Fit("crossing", -0.7, 0.015), 1024, CONFIG) == "target"
    assert break_even.stop_reason(break_even.Fit("crossing", -0.7, 0.03), 1024, CONFIG) is None
    assert break_even.stop_reason(break_even.Fit("crossing", -0.7, 0.03), 8192, CONFIG) == "budget"
    floor = break_even.Fit("separated", lower=None, upper=(1, 20))
    ceiling = break_even.Fit("separated", lower=(20, 1), upper=None)
    assert break_even.stop_reason(floor, 64, CONFIG) == "bound"
    assert break_even.stop_reason(ceiling, 64, CONFIG) == "bound"
    floored = dataclasses.replace(CONFIG, minimum_count=3)
    assert break_even.stop_reason(break_even.Fit("separated", upper=(3, 20)), 64, floored) == (
        "bound"
    )
    open_bound = break_even.Fit("separated", lower=None, upper=(6, 20))
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


SMALL = break_even.CrossoverConfig(fixed_count=3, minimum_count=1, bracket=(1, 3), budget_games=32)


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
    assert all(
        (block["policy_count"], block["opponent_count"]) == (3, 3) for block in lines[0]["blocks"]
    )
    assert all(
        block["wins"] + block["draws"] + block["losses"] == LAYOUT.block_environments
        and max(block["policy_count"], block["opponent_count"]) == 3
        for line in lines
        for block in line["blocks"]
    )
    assert sum(row["games"] for row in first["lineups"]) == first["games"]
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
def test_the_opponents_ships_shrink_after_a_policy_loses_at_parity(tmp_path):
    """A recorded parity batch lost outright: the next batch doubles the ratio by benching
    the opponent's ships (3 against 2), the policy keeping its 3."""
    record = tmp_path / "r.jsonl"
    key = break_even.record_key(_scripted(), _scripted(), SHORT, LAYOUT.seed, SMALL)
    lost = dict(policy_count=3, opponent_count=3, wins=0, draws=0, losses=8)
    record.write_text(json.dumps(dict(key=key, batch=0, blocks=[lost, lost])) + "\n")
    found = break_even.measure(
        _scripted(), _scripted(), SHORT, CPU, LAYOUT, SMALL, record, lambda text: None
    )
    second = json.loads(record.read_text().splitlines()[1])
    assert second["batch"] == 1
    assert [(one["policy_count"], one["opponent_count"]) for one in second["blocks"]] == [
        (3, 2),
        (3, 2),
    ]
    assert found["games"] == 32
    assert {(row["policy_count"], row["opponent_count"]) for row in found["lineups"]} == {
        (3, 3),
        (3, 2),
    }


def test_the_break_even_game_keeps_the_presets_arena_unless_the_density_rule_is_asked():
    base = api.PRESETS["real_scale_deathmatch"]
    config = break_even.CrossoverConfig()
    fixed = break_even.break_even_game(base, config)
    assert (fixed.ships_per_team, fixed.arena_radius) == (20, base.arena_radius)
    density = break_even.break_even_game(base, config, "density")
    assert density.ships_per_team == 20
    assert density.arena_radius == api.arena_radius_for(base, config.fixed_count)
    assert break_even.sized_game(base, base.ships_per_team, "density") == base
    with pytest.raises(ValueError, match="map rule"):
        break_even.sized_game(base, 3, "scaled")


def test_a_record_of_another_game_version_or_floor_is_another_measurement(tmp_path):
    """The key names the game version and the lineups' range: other batches are not
    reused."""
    from bnb.game import VERSION

    record = tmp_path / "small.batches.jsonl"
    key = break_even.record_key(_scripted(), _scripted(), SHORT, 0, SMALL)
    assert key["game_version"] == VERSION
    assert (key["fixed_count"], key["minimum_count"]) == (3, 1)
    older = dict(key, game_version=VERSION - 1)
    record.write_text(json.dumps(dict(key=older, batch=0)) + "\n")
    assert break_even.stored_batches(record, key) == []
    assert len(break_even.stored_batches(record, older)) == 1
    floored = dataclasses.replace(SMALL, minimum_count=2)
    assert break_even.record_key(_scripted(), _scripted(), SHORT, 0, floored) != key


@pytest.mark.usefixtures("global_generator")
def test_a_network_plays_a_measurement(tmp_path):
    game = GameConfig(2, 2, 700.0, episode_seconds=4.0, respawn_seconds=1.0)
    directory = tmp_path / "policy"
    checkpoint.save_policy(directory, tiny_model("yemong", seed=1, game=game))
    layout = dataclasses.replace(LAYOUT, block_environments=4)
    config = break_even.CrossoverConfig(
        fixed_count=2, minimum_count=1, bracket=(1, 2), budget_games=8
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


def test_the_game_must_be_the_fixed_sides_size():
    with pytest.raises(ValueError, match="ships per team"):
        break_even.measure(
            _scripted(), _scripted(), GAME, CPU, LAYOUT, CONFIG, None, lambda text: None
        )


# ---- the command


def test_the_command_parses_its_options_with_the_decided_defaults():
    import tyro

    from bnb.cli import COMMANDS, Crossover

    options = tyro.cli(Crossover, args=["policy"])
    assert options.policy == "policy" and options.opponent == "imugi"
    assert (options.fixed_count, options.minimum_count) == (20, 1)
    assert options.map_rule == "preset"
    assert options.presets == (
        "real_scale_frontline",
        "real_scale_deathmatch",
        "real_scale_elimination",
    )
    assert options.lineup_types == ("Yongma", "Jujak") and options.target_stderr == 0.02
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
        fixed_count=3,
        minimum_count=1,
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
    assert (
        "tiny (1 of 1): scripted against scripted, the weaker side at 3 ships and the "
        "stronger from 1, 3v3 game" in printed
    )
    assert f"arena radius {SHORT.arena_radius:g} m (preset)" in printed
    record = tmp_path / "artifacts" / "crossover" / "tiny.batches.jsonl"
    assert len(record.read_text().splitlines()) == 1
    (directory,) = [
        path for path in (tmp_path / "artifacts" / "crossover").iterdir() if path.is_dir()
    ]
    result = json.loads((directory / "result.json").read_text())
    assert result["payload"]["presets"]["tiny"]["games"] == 16
    assert result["payload"]["lineup_types"] == list(names[:2])
    assert "command" in result["provenance"]
