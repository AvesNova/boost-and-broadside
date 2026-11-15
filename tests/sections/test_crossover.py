"""The break-even lineup (`bnb.league.crossover`, `bnb crossover`): the lineups on the log
ratio axis, the fit recovers a known crossing on either side of parity and gives a bound on
separated outcomes, the allocation starts at parity, picks the shrinking side from the games
and concentrates the games near the crossing, the stop rule, the per-game lineups, the
engine's benching, and a small measurement on the CPU that records every batch and resumes
from its record."""

import collections
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


def test_the_lineups_run_along_the_ratio_with_the_weaker_side_fixed_down_to_one_ship():
    config = break_even.CrossoverConfig()
    lineups = config.lineups()
    ratios = [break_even.log_ratio(lineup) for lineup in lineups]
    assert ratios == sorted(ratios) and len(set(ratios)) == len(lineups) == 2 * 19 + 1
    assert lineups[0] == (1, 20) and lineups[-1] == (20, 1) and (20, 20) in lineups
    assert all(20 in lineup and min(lineup) >= 1 and max(lineup) <= 20 for lineup in lineups)
    assert config.ships_per_team == 20
    clipped = break_even.CrossoverConfig(bracket=(0, 10, 30))
    assert clipped.bracket_lineups("policy") == [(1, 20), (10, 20), (20, 20)]
    assert clipped.bracket_lineups("opponent") == [(20, 20), (20, 10), (20, 1)]
    assert len(clipped.bracket_lineups("both")) == 5
    with pytest.raises(ValueError, match="fixed count"):
        break_even.CrossoverConfig(fixed_count=0)


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
    # the reported ratio is the opponent's ships per one of the policy's
    ratio, ratio_stderr = found.ratio()
    assert ratio == pytest.approx(1 / 1.2, rel=1e-6)
    assert ratio_stderr == pytest.approx(ratio * found.stderr)
    assert found.lineup(20) == pytest.approx((20.0, 20 / 1.2))
    stronger = dataclasses.replace(found, break_even=math.log(0.45))
    assert stronger.lineup(20) == pytest.approx((9.0, 20.0))
    assert stronger.ratio()[0] == pytest.approx(20 / 9)  # holds 2.22 times its number
    assert not stronger.suspicious(20) and not found.suspicious(20)


@pytest.mark.parametrize(
    ("policy_per_opponent", "suspicious"),
    [(1.5 / 20, True), (2 / 20, True), (2.5 / 20, False), (20 / 2, True), (20 / 3, False)],
)
def test_a_break_even_at_two_ships_or_fewer_is_suspicious(policy_per_opponent, suspicious):
    found = break_even.Fit("crossing", math.log(policy_per_opponent), 0.01)
    assert found.suspicious(20) is suspicious
    assert not break_even.Fit("separated", upper=(1, 20)).suspicious(20)


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


def _paired(lineups) -> list:
    """Each lineup for a pair of games, one in each seat."""
    return [lineup for lineup in lineups for _ in range(2)]


def _games(true_ratio: float, slope: float, lineups, generator) -> list[dict]:
    """`_blocks` for a batch's per-game lineups: one block per lineup with its games."""
    counts = collections.Counter(lineups)
    played = sorted(counts, key=break_even.log_ratio)
    return [
        block
        for lineup in played
        for block in _blocks(true_ratio, slope, [lineup], counts[lineup], generator)
    ]


def test_the_first_batch_spreads_over_the_whole_axis():
    first = break_even.first_lineups(CONFIG)
    assert len(first) == 13 and first[0] == (1, 20) and first[-1] == (20, 1)
    assert (20, 20) in first
    assert sum(one[0] < one[1] for one in first) == sum(one[0] > one[1] for one in first) == 6
    ratios = [break_even.log_ratio(one) for one in first]
    steps = [later - earlier for earlier, later in zip(ratios, ratios[1:], strict=False)]
    assert max(steps) < 2.5 * min(steps)  # about even in r, as the integer lineups allow
    lineups = break_even.allocate(break_even.Fit("empty"), {}, CONFIG, 52)
    assert lineups == _paired(first * 2)
    small = break_even.CrossoverConfig(fixed_count=3)
    assert break_even.first_lineups(small) == [(1, 3), (2, 3), (3, 3), (3, 2), (3, 1)]
    with pytest.raises(ValueError, match="even"):
        break_even.allocate(break_even.Fit("empty"), {}, CONFIG, 3)


def test_without_a_fit_the_games_pick_the_shrinking_side():
    winning = _rows((20, 20, 6, 0, 4))
    found = break_even.fit(winning)
    assert break_even.allocate(found, winning, CONFIG, 8) == _paired(
        [(6, 20), (10, 20), (14, 20), (20, 20)]
    )
    losing = _rows((20, 20, 3, 2, 5))
    assert break_even.allocate(break_even.fit(losing), losing, CONFIG, 8) == _paired(
        [(20, 20), (20, 14), (20, 10), (20, 6)]
    )
    even = _rows((20, 20, 4, 2, 4))
    lineups = break_even.allocate(break_even.fit(even), even, CONFIG, 14)
    assert sorted(set(lineups), key=break_even.log_ratio) == CONFIG.bracket_lineups("both")


def test_the_allocation_concentrates_games_at_the_crossing_in_seat_pairs():
    tallies = _synthetic(0.4, 5.0, [(6, 20), (10, 20), (14, 20), (20, 20)], 256, seed=1)
    found = break_even.fit(tallies)
    lineups = break_even.allocate(found, tallies, CONFIG, 64)
    assert len(lineups) == 64 and lineups[0::2] == lineups[1::2]
    assert all(abs(break_even.log_ratio(one) - found.break_even) < 0.3 for one in lineups)


@pytest.mark.parametrize("true_ratio", [9.3 / 20, 20 / 13, 20 / 19.5, 0.2, 20 / 3])
def test_a_simulated_measurement_finds_the_ratio_on_either_side_of_parity(true_ratio):
    """Synthetic games from a known logistic in the log ratio, stronger policies', weaker
    ones' and one near parity, played batch by batch through the allocation from the first
    (whole-axis) batch: the fit finds the ratio within its standard error, most games go
    near the crossing, and no lineup has more than 20 a side or fewer than 1."""
    generator = torch.Generator().manual_seed(3)
    blocks: list[dict] = []
    found, tallies = break_even.Fit("empty"), {}
    for _ in range(12):
        lineups = break_even.allocate(found, tallies, CONFIG, 512)
        blocks += _games(true_ratio, 5.0, lineups, generator)
        tallies = break_even.tallies(blocks)
        found = break_even.fit(tallies)
    assert found.status == "crossing"
    assert abs(found.break_even - math.log(true_ratio)) < 3 * found.stderr
    assert found.stderr < 0.05
    played = {(one["policy_count"], one["opponent_count"]) for one in blocks}
    assert all(max(one) <= 20 and min(one) >= 1 and 20 in one for one in played)
    assert not found.suspicious(20)
    games = sum(one["wins"] + one["draws"] + one["losses"] for one in blocks)
    near = sum(
        one["wins"] + one["draws"] + one["losses"]
        for one in blocks
        if abs(
            break_even.log_ratio((one["policy_count"], one["opponent_count"])) - found.break_even
        )
        < 0.3
    )
    assert games == 12 * 512 and near / games > 0.6


def test_separated_outcomes_steer_into_or_beyond_the_bound():
    wins = break_even.Fit("separated", lower=None, upper=(6, 20))
    assert break_even.allocate(wins, {}, CONFIG, 4) == [(3, 20)] * 4
    losses = break_even.Fit("separated", lower=(6, 20), upper=None)
    assert break_even.allocate(losses, {}, CONFIG, 2) == [(12, 20)] * 2
    lost_at_parity = break_even.Fit("separated", lower=(20, 20), upper=None)
    assert break_even.allocate(lost_at_parity, {}, CONFIG, 2) == [(20, 10)] * 2
    won_at_parity = break_even.Fit("separated", lower=None, upper=(20, 20))
    assert break_even.allocate(won_at_parity, {}, CONFIG, 2) == [(10, 20)] * 2
    inside = break_even.Fit("separated", lower=(6, 20), upper=(14, 20))
    assert set(break_even.allocate(inside, {}, CONFIG, 6)) == {(8, 20), (9, 20), (10, 20)}
    across = break_even.Fit("separated", lower=(18, 20), upper=(20, 18))
    assert set(break_even.allocate(across, {}, CONFIG, 6)) == {(19, 20), (20, 20), (20, 19)}
    adjacent = break_even.Fit("separated", lower=(9, 20), upper=(10, 20))
    assert break_even.allocate(adjacent, {}, CONFIG, 8) == _paired([(9, 20), (10, 20)] * 2)
    near_floor = break_even.Fit("separated", lower=None, upper=(2, 20))
    assert break_even.allocate(near_floor, {}, CONFIG, 2) == [(1, 20)] * 2
    # a mixed lineup past the last all-lost one: the next doubles from the mixed one
    mixed = _rows((20, 20, 0, 0, 8), (20, 10, 0, 0, 8), (20, 5, 1, 0, 7))
    stuck = break_even.fit(mixed)
    assert (stuck.status, stuck.lower, stuck.upper) == ("separated", (20, 10), None)
    assert break_even.allocate(stuck, mixed, CONFIG, 2) == [(20, 3)] * 2  # r nearest log 8


def test_the_stop_rule():
    assert break_even.stop_reason(break_even.Fit("crossing", -0.7, 0.015), 1024, CONFIG) == "target"
    assert break_even.stop_reason(break_even.Fit("crossing", -0.7, 0.03), 1024, CONFIG) is None
    assert break_even.stop_reason(break_even.Fit("crossing", -0.7, 0.03), 8192, CONFIG) == "budget"
    floor = break_even.Fit("separated", lower=None, upper=(1, 20))
    ceiling = break_even.Fit("separated", lower=(20, 1), upper=None)
    assert break_even.stop_reason(floor, 64, CONFIG) == "bound"
    assert break_even.stop_reason(ceiling, 64, CONFIG) == "bound"
    assert break_even.stop_reason(break_even.Fit("separated", upper=(2, 20)), 64, CONFIG) is None
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
    arena.play([(0, 1)] * LAYOUT.blocks, _by_block([(3, 3), (2, 3)]), lineups)
    assert torch.equal(arena.rollout.ship_type.cpu(), lineups.repeat(1, 2))
    with pytest.raises(ValueError, match="per-game lineups"):
        arena.play([(0, 1)] * LAYOUT.blocks, _by_block([(3, 3), (2, 3)]))
    with pytest.raises(ValueError, match="per-game lineups"):
        Arena(
            [_scripted(), Player("typed", rung=Rung(1.0), ship_type=0)],
            GAME,
            CPU,
            LAYOUT,
            game_lineups=True,
        )


def _by_block(pairs) -> list:
    """Per-game active counts with every game of block b at pairs[b]."""
    return [pair for pair in pairs for _ in range(LAYOUT.block_environments)]


# several lineups in one batch, one pair of games (both seats) each, in both blocks
MIXED = [(3, 3), (3, 3), (1, 3), (1, 3), (3, 1), (3, 1), (2, 3), (2, 3)] * 2


@pytest.mark.usefixtures("global_generator")
def test_each_environment_gives_the_first_player_its_count_whatever_its_seat():
    arena = Arena([_scripted(), _scripted()], GAME, CPU, LAYOUT, benching=True)
    active = arena._active(MIXED)  # (B, 2) by team
    first_team = arena.seats.long()
    first = active.gather(1, first_team[:, None])[:, 0]
    second = active.gather(1, 1 - first_team[:, None])[:, 0]
    assert first.tolist() == [pair[0] for pair in MIXED]
    assert second.tolist() == [pair[1] for pair in MIXED]
    assert set(first_team.tolist()) == {0, 1}  # both seats in every block
    with pytest.raises(ValueError, match="from 1 to"):
        arena.play([(0, 1), (0, 1)], [(4, 3)] + MIXED[1:])
    with pytest.raises(ValueError, match="one pair per game"):
        arena.play([(0, 1), (0, 1)], [(3, 3), (3, 3)])


@pytest.mark.usefixtures("global_generator")
def test_one_batch_plays_several_lineups_and_tallies_each_game():
    """Per-game lineups in one batch: each game's outcome is kept, the blocks' sums are
    `play`'s, and `block_tallies` groups them per block and lineup."""
    arena = Arena([_scripted(), _scripted()], SHORT, CPU, LAYOUT, benching=True, game_lineups=True)
    types = break_even.lineup_draw(0, 0, LAYOUT.games, SHORT.ships_per_team, (0, 1))
    by_block = arena.play([(0, 1)] * LAYOUT.blocks, MIXED, types)
    outcomes = arena.game_outcomes()
    assert outcomes.shape == (LAYOUT.games, 3) and outcomes.sum(-1).tolist() == [1] * 16
    assert torch.equal(outcomes.reshape(LAYOUT.blocks, -1, 3).sum(1), by_block)
    entries = break_even.block_tallies(MIXED, outcomes, LAYOUT.block_environments)
    assert [(one["block"], one["policy_count"], one["opponent_count"]) for one in entries] == [
        (block, *lineup) for block in (0, 1) for lineup in ((1, 3), (2, 3), (3, 3), (3, 1))
    ]
    assert all(one["wins"] + one["draws"] + one["losses"] == 2 for one in entries)


@pytest.mark.usefixtures("global_generator")
def test_the_rollout_shows_the_benched_flag_from_its_first_observation_on():
    """The restart's observation (before any tick) and every one after it flag exactly the
    benched rows of both views, as the rollout's active counts give them."""
    arena = Arena([_scripted(), _scripted()], GAME, CPU, LAYOUT, benching=True)
    rollout = arena.rollout
    rollout.restart(arena._active(MIXED))
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
        arena.play([(0, 1), (0, 1)], _by_block([(3, 3), (3, 3)]))


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


SMALL = break_even.CrossoverConfig(fixed_count=3, bracket=(1, 3), budget_games=32)


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
    # the first batch spreads over the whole axis, both sides of parity in one batch
    first_lineups = {(one["policy_count"], one["opponent_count"]) for one in lines[0]["blocks"]}
    assert first_lineups == set(break_even.first_lineups(SMALL))
    assert {(1, 3), (3, 1)} <= first_lineups
    for line in lines:
        assert all(max(one["policy_count"], one["opponent_count"]) == 3 for one in line["blocks"])
        for block in range(LAYOUT.blocks):
            mine = [one for one in line["blocks"] if one["block"] == block]
            games = sum(one["wins"] + one["draws"] + one["losses"] for one in mine)
            assert games == LAYOUT.block_environments
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


def test_a_suspicious_break_even_is_recorded_and_warned_about(tmp_path):
    """A recorded measurement crossing at 1.5 ships against 20 (at its budget, so nothing is
    played): the result carries the flag and the progress prints the warning."""
    game = dataclasses.replace(SHORT, ships_per_team=20)
    config = dataclasses.replace(CONFIG, budget_games=64)
    record = tmp_path / "r.jsonl"
    key = break_even.record_key(_scripted(), _scripted(), game, LAYOUT.seed, config)
    blocks = []
    for lineup in ((1, 20), (2, 20), (3, 20)):
        p = _curve(1.5 / 20, 3.0, lineup)
        blocks.append(
            dict(
                policy_count=lineup[0],
                opponent_count=lineup[1],
                wins=100 * p,
                draws=0,
                losses=100 * (1 - p),
            )
        )
    record.write_text(json.dumps(dict(key=key, batch=0, blocks=blocks)) + "\n")
    printed: list[str] = []
    found = break_even.measure(
        _scripted(), _scripted(), game, CPU, LAYOUT, config, record, printed.append
    )
    assert found["stop"] == "budget" and found["suspicious"]
    assert found["ratio"] == pytest.approx(20 / 1.5, rel=1e-4)
    assert found["lineup"] == pytest.approx([1.5, 20.0], rel=1e-4)
    assert any(text.startswith("WARNING") and "bug" in text for text in printed)


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


def test_a_record_of_another_game_version_or_fixed_count_is_another_measurement(tmp_path):
    """The key names the game version and the fixed count: other batches are not
    reused."""
    from bnb.game import VERSION

    record = tmp_path / "small.batches.jsonl"
    key = break_even.record_key(_scripted(), _scripted(), SHORT, 0, SMALL)
    assert key["game_version"] == VERSION
    assert key["fixed_count"] == 3
    older = dict(key, game_version=VERSION - 1)
    record.write_text(json.dumps(dict(key=older, batch=0)) + "\n")
    assert break_even.stored_batches(record, key) == []
    assert len(break_even.stored_batches(record, older)) == 1
    smaller = dataclasses.replace(SMALL, fixed_count=2)
    assert break_even.record_key(_scripted(), _scripted(), SHORT, 0, smaller) != key


@pytest.mark.usefixtures("global_generator")
def test_a_network_plays_a_measurement(tmp_path):
    game = GameConfig(2, 2, 700.0, episode_seconds=4.0, respawn_seconds=1.0)
    directory = tmp_path / "policy"
    checkpoint.save_policy(directory, tiny_model("yemong", seed=1, game=game))
    layout = dataclasses.replace(LAYOUT, block_environments=4)
    config = break_even.CrossoverConfig(fixed_count=2, bracket=(1, 2), budget_games=8)
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
    assert options.fixed_count == 20 and not hasattr(options, "minimum_count")
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
