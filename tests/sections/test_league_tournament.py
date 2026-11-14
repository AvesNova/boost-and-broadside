"""Published ratings and head-to-head (league spec 6): the match engine on fixed slots, the
allocation of its blocks, the allowed pairs, the tournament's anchor, stopping rule and
recovery of a known field, and head-to-head's report, on a small 2v2 game on the CPU."""

import dataclasses
import datetime
import json
import math

import pytest
import torch
from torch.utils import _pytree

from bnb.contracts import GameConfig
from bnb.league import tournament
from bnb.league.matches import SEAT, Arena, ArenaConfig, Player, default_lanes, player
from bnb.league.ratings import win_probability
from bnb.models.relational import Relational, RelationalBuild, RelationalConfig
from bnb.models.yemong import policy
from bnb.runs import checkpoint
from bnb.runtime.opponents import Rung

pytestmark = pytest.mark.usefixtures("global_generator")

GAME = GameConfig(2, 2, 1500.0, episode_seconds=2.0, respawn_seconds=1.0)
CPU = torch.device("cpu")
SCRIPTED = Player("scripted", rung=Rung(1.0))
RANDOM = Player("random", rung=Rung(math.inf))
HALF = Player("rung_T2", rung=Rung(2.0))
SMALL = ArenaConfig(blocks=2, block_environments=4, length=16, compiled=False)


def _network(directory, seed: int, hidden: int = 16) -> Player:
    """A policy file of a small relational model with random weights throughout (its
    initial policy head is zero, so every such model would play alike)."""
    torch.manual_seed(seed)
    model = Relational(RelationalBuild(RelationalConfig(hidden=hidden), 2, 4))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_()
    checkpoint.save_policy(directory, model)
    return player(str(directory))


def _block_state(arena: Arena, block: int) -> list[torch.Tensor]:
    """Every per-environment leaf of the game state and of the batch's sums (shots, hits,
    rewards, ...), for one block's environments."""
    environments = arena.config.block_environments
    rows = slice(block * environments, (block + 1) * environments)
    carry = arena.rollout.carry
    leaves = _pytree.tree_leaves((carry["state"], carry["game_sums"]))
    return [
        leaf[rows].clone()
        for leaf in leaves
        if isinstance(leaf, torch.Tensor) and leaf.dim() and leaf.shape[0] == arena.config.games
    ]


# ---- the engine


def test_ladder_names_parse_and_others_do_not():
    assert player("random") == RANDOM and player("scripted") == SCRIPTED
    assert player("rung_T2") == HALF
    with pytest.raises(ValueError, match="not a policy file"):
        player("no-such-player")


def test_every_block_plays_each_environments_first_game_once_with_the_seats_split():
    arena = Arena([SCRIPTED, RANDOM, HALF], GAME, CPU, SMALL)
    for matchups in ([(0, 1), (2, 1)], [(1, 2), (0, 2)]):
        results = arena.play(matchups)
        assert results.sum(1).tolist() == [4, 4]
    # each block's first seat's rows, then its second's; environment i seats the first
    # player on team i mod 2
    assert arena.rollout.lineup.role[SEAT].tolist() == ([0] * 4 + [1] * 4) * 2
    assert arena.seats.tolist() == [0, 1] * 4


def test_the_layout_needs_even_blocks_and_a_matchup_per_block():
    with pytest.raises(ValueError, match="even"):
        ArenaConfig(block_environments=3)
    arena = Arena([SCRIPTED, RANDOM], GAME, CPU, SMALL)
    with pytest.raises(ValueError, match="blocks"):
        arena.play([(0, 1)])
    with pytest.raises(ValueError, match="itself"):
        arena.play([(0, 1), (1, 1)])


def test_slots_take_their_players_in_place(tmp_path):
    first, second = _network(tmp_path / "a", 1), _network(tmp_path / "b", 2)
    arena = Arena([first, second, SCRIPTED, HALF], GAME, CPU, SMALL)
    seat = arena.seat
    buffers = {name: value.data_ptr() for name, value in seat.parameters[0].items()}
    arena.play([(0, 2), (3, 1)])
    name = next(iter(arena.states[0]))
    stacked = seat.parameters[0][f"model.{name}"]  # by lane: the slots in order
    assert seat.lanes == (4,) and seat.lane_of == {0: (0, 0), 3: (0, 1)}
    assert seat.lane_slot[0][:2].tolist() == [0, 3] and seat.lane_active[0].tolist() == [
        True,
        True,
        False,
        False,
    ]
    assert torch.equal(stacked[0], arena.states[0][name])  # block 0's first seat
    assert torch.equal(stacked[1], arena.states[1][name])  # block 1's second seat
    assert sorted(seat.lane_slot[0].tolist()) == [0, 1, 2, 3]  # the lanes on different slots
    rows = seat.slot_rows
    assert seat.scripted.tolist() == [False] * rows + [True] * 2 * rows + [False] * rows
    assert seat.temperature[2 * rows : 3 * rows].tolist() == [2.0] * rows  # rung_T2
    arena.play([(1, 0), (2, 3)])
    assert torch.equal(stacked[0], arena.states[1][name])
    assert torch.equal(stacked[1], arena.states[0][name])
    assert seat.lane_slot[0][:2].tolist() == [0, 1] and seat.lane_of == {0: (0, 0), 1: (0, 1)}
    assert sorted(seat.lane_slot[0].tolist()) == [0, 1, 2, 3]
    assert seat.scripted.tolist() == [False] * 2 * rows + [True] * 2 * rows
    # every swap is a copy into the same buffers (a captured graph keeps reading them)
    assert {name: value.data_ptr() for name, value in seat.parameters[0].items()} == buffers


def test_a_block_plays_the_same_games_whatever_the_other_blocks_play(tmp_path):
    """Each slot's network acts on its own rows only: block 0's games are the same whether
    block 1 plays two networks or two scripted players."""
    first, second = _network(tmp_path / "a", 1), _network(tmp_path / "b", 2)
    arena = Arena([first, second, SCRIPTED, RANDOM], GAME, CPU, SMALL)
    states = []
    for other in ((1, 0), (2, 3)):
        torch.manual_seed(5)
        arena.play([(0, 3), other])
        states.append(_block_state(arena, 0))
    assert states[0] and len(states[0]) == len(states[1])
    for one, two in zip(*states, strict=True):
        assert torch.equal(one, two)
    torch.manual_seed(5)
    arena.play([(1, 3), (1, 0)])  # another network in block 0 plays other games
    assert any(not torch.equal(one, two) for one, two in zip(states[0], _block_state(arena, 0)))


def test_networks_of_different_builds_are_grouped_by_build(tmp_path):
    first, second = _network(tmp_path / "a", 1), _network(tmp_path / "b", 2)
    wider = _network(tmp_path / "c", 3, hidden=32)
    arena = Arena([first, wider, SCRIPTED, second], GAME, CPU, SMALL, lanes=(2, 2))
    assert len(arena.groups) == 2 and len(arena.seat.parameters) == 2
    assert arena.group_of == {0: 0, 1: 1, 3: 0}  # the groups in order of first appearance
    arena.play([(0, 1), (3, 1)])  # a block's two seats from different groups
    assert arena.seat.lane_of == {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}
    name = next(iter(arena.states[1]))
    assert torch.equal(arena.seat.parameters[1][f"model.{name}"][1], arena.states[1][name])
    # by default the lanes split the slots as the groups' players: 2 and 1 of 4 slots
    assert Arena([first, wider, SCRIPTED, second], GAME, CPU, SMALL).lanes == (3, 1)
    with pytest.raises(ValueError, match="1 lanes"):
        Arena([first, wider, SCRIPTED, second], GAME, CPU, SMALL).play([(0, 1), (3, 1)])


def test_a_two_build_field_plays_and_a_block_does_not_depend_on_the_others(tmp_path):
    first, wider = _network(tmp_path / "a", 1), _network(tmp_path / "b", 2, hidden=32)
    arena = Arena([first, wider, SCRIPTED, RANDOM], GAME, CPU, SMALL)
    results = arena.play([(0, 1), (1, 2)])
    assert results.sum(1).tolist() == [4, 4]
    states = []
    for other in ((1, 0), (2, 3)):
        torch.manual_seed(5)
        arena.play([(0, 3), other])
        states.append(_block_state(arena, 0))
    for one, two in zip(*states, strict=True):
        assert torch.equal(one, two)


def _greedy(logits):
    """The policy's most likely joint action, drawing nothing: a network's play then depends
    on its rows alone, whatever the other lanes draw."""
    joint = logits.argmax(-1)
    return (joint.to(torch.int32), *policy.evaluate(logits, joint))


def test_a_mixed_field_plays_each_pair_as_the_pair_alone(tmp_path, monkeypatch):
    """Each row runs its own player's network once, on its group's lane: with greedy
    networks, a block of a three-build field plays the games the same pair plays in an
    engine of its two builds alone, block for block (the scripted draws are the same)."""
    monkeypatch.setattr(policy, "sample", _greedy)
    first, second = _network(tmp_path / "a", 1), _network(tmp_path / "b", 2, hidden=24)
    third = _network(tmp_path / "c", 3, hidden=32)
    mixed = Arena([first, second, third, SCRIPTED, RANDOM], GAME, CPU, SMALL)
    alone = Arena([first, second, SCRIPTED, RANDOM], GAME, CPU, SMALL)
    assert len(mixed.groups) == 3 and mixed.lanes == (2, 1, 1)
    assert len(alone.groups) == 2 and alone.lanes == (2, 2)
    for matchups, versus in (
        ([(0, 1), (2, 3)], [(0, 1), (2, 3)]),  # block 1: the third build, or scripted
        ([(1, 0), (4, 2)], [(1, 0), (3, 2)]),  # the seats swapped; random against a network
    ):
        states, results = [], []
        for arena, pairs in ((mixed, matchups), (alone, versus)):
            torch.manual_seed(5)
            results.append(arena.play(pairs))
            states.append(_block_state(arena, 0))
        assert torch.equal(results[0][0], results[1][0])
        assert states[0] and len(states[0]) == len(states[1])
        for one, two in zip(*states, strict=True):
            assert torch.equal(one, two)


def test_the_lanes_split_the_slots_as_the_groups_players():
    # one network player fills at most one slot of each of the 8 blocks
    assert default_lanes([None, 0, None], 16) == (8,)
    assert default_lanes([None, 0, 0], 16) == (16,)
    assert default_lanes([0, 1, None], 16) == (8, 8)
    assert default_lanes([None, 0, 1, 1, 2], 16) == (4, 8, 4)
    assert default_lanes([0, 1, 2, 3, 4], 16) == (4, 3, 3, 3, 3)
    assert sum(default_lanes([0, 1, 1, 1, 2, 3, 4, None], 16)) == 16
    assert default_lanes([None, None], 16) == ()
    with pytest.raises(ValueError, match="more than 2 slots"):
        default_lanes([0, 1, 2], 2)


def test_a_batch_keeps_to_the_engines_lanes(tmp_path):
    first, wider = _network(tmp_path / "a", 1), _network(tmp_path / "b", 2, hidden=32)
    arena = Arena([first, wider, SCRIPTED], GAME, CPU, SMALL, lanes=(1, 3))
    with pytest.raises(ValueError, match="2 slots of network group 0, which has 1 lanes"):
        arena.play([(0, 2), (0, 1)])
    assert arena.play([(0, 1), (1, 2)]).sum(1).tolist() == [4, 4]


def test_the_tournament_rates_a_two_build_field(tmp_path):
    first, wider = _network(tmp_path / "a", 1), _network(tmp_path / "b", 2, hidden=32)
    config = tournament.TournamentConfig(SMALL, target_stderr=1.0, hopeless=math.inf, budget=16)
    rows = tournament.rate([SCRIPTED, first, wider], "small", CPU, config, game=GAME)
    by_name = {row["player"]: row for row in rows}
    assert set(by_name) == {"scripted", str(tmp_path / "a"), str(tmp_path / "b")}
    assert all(math.isfinite(row["calibrated_elo"]) and row["games"] > 0 for row in rows)


def test_networks_of_one_build_are_one_group(tmp_path):
    first, second = _network(tmp_path / "a", 1), _network(tmp_path / "b", 2)
    arena = Arena([first, second, SCRIPTED], GAME, CPU, SMALL)
    assert len(arena.groups) == 1 and arena.group_of == {0: 0, 1: 0}
    assert len(Arena([SCRIPTED, RANDOM], GAME, CPU, SMALL).groups) == 0


def test_scripted_at_temperature_one_and_floor_zero_plays_exactly_as_scripted():
    identity = player("scripted@T=1,floor=0")
    states = []
    for one in (SCRIPTED, identity):
        arena = Arena([one, HALF], GAME, CPU, SMALL)
        torch.manual_seed(7)
        arena.play([(0, 1), (1, 0)])
        states.append(_block_state(arena, 0) + _block_state(arena, 1))
    for one, two in zip(*states, strict=True):
        assert torch.equal(one, two)


def test_a_teacher_takes_its_temperature_and_floor():
    softened = player("scripted@T=4,floor=0.5")
    arena = Arena([softened, SCRIPTED], GAME, CPU, SMALL)
    assert int(arena.play([(0, 1), (1, 0)]).sum()) == 8
    rows = arena.seat.slot_rows
    assert arena.seat.floor[:rows].tolist() == [0.5] * rows
    assert arena.seat.temperature[:rows].tolist() == [4.0] * rows
    assert arena.seat.floor[rows : 3 * rows].tolist() == [0.0] * 2 * rows


# ---- ablated players


def _yemong(directory) -> Player:
    """A policy file of a Yemong for the small game with random last layers (so its play
    depends on its inputs)."""
    from bnb.game import api
    from bnb.models.yemong.model import Yemong, YemongConfig, yemong_build

    with torch.no_grad():
        small = YemongConfig(width=64, heads=1, map_width=32, blocks=1)
        model = Yemong(yemong_build(small, api.rules_of(GAME), api.ship_types()))
        for name, parameter in model.named_parameters():
            if name.endswith("last.weight") or ".lasts." in name and name.endswith(".weight"):
                parameter.normal_(0.0, 0.05)
    checkpoint.save_policy(directory, model)
    return player(str(directory))


def test_an_ablated_name_is_its_policy_with_sorted_ablations(tmp_path):
    base = _yemong(tmp_path / "a")
    rotary = player(f"{base.name}:rotary")
    assert rotary == Player(f"{base.name}:rotary", policy=base.policy, ablations=("rotary",))
    assert player(f"{base.name}:rotary+forecast").ablations == ("forecast", "rotary")
    with pytest.raises(ValueError, match="twice"):
        player(f"{base.name}:rotary+rotary")
    with pytest.raises(ValueError, match="only a network"):
        player("scripted:rotary")
    with pytest.raises(ValueError, match="not a policy file"):
        player(f"{base.name}:sideways")


def test_only_a_yemong_plays_ablated(tmp_path):
    relational = _network(tmp_path / "a", 1)
    ablated = player(f"{relational.name}:rotary")
    with pytest.raises(ValueError, match="Yemong"):
        Arena([relational, ablated, SCRIPTED], GAME, CPU, SMALL)


def test_an_ablated_copy_shares_its_weights_and_ablates_only_its_slots_rows(tmp_path):
    base = _yemong(tmp_path / "a")
    rotary, forecast = player(f"{base.name}:rotary"), player(f"{base.name}:forecast")
    arena = Arena([base, rotary, forecast, SCRIPTED], GAME, CPU, SMALL)
    assert arena.states[0] is arena.states[1] is arena.states[2]
    arena.play([(1, 3), (0, 2)])
    rows = arena.seat.slot_rows
    ablations = arena.seat.ablations
    assert ablations.rotary.tolist() == [True] * rows + [False] * 3 * rows
    assert ablations.forecast.tolist() == [False] * 3 * rows + [True] * rows
    arena.play([(3, 2), (1, 0)])
    assert ablations.rotary.tolist() == [False] * 2 * rows + [True] * rows + [False] * rows
    assert ablations.forecast.tolist() == [False] * rows + [True] * rows + [False] * 2 * rows


def test_a_two_build_field_with_an_ablated_yemong(tmp_path):
    """A relational network beside a Yemong and its ablated copy: the copy shares the
    Yemong's group and weights, only its slot's rows are ablated, the relational group is
    called without ablations, and a relational block plays the same games as in an engine
    without the ablated copy."""
    relational, yemong = _network(tmp_path / "a", 1), _yemong(tmp_path / "b")
    rotary = player(f"{yemong.name}:rotary")
    arena = Arena([relational, yemong, rotary, SCRIPTED], GAME, CPU, SMALL)
    assert len(arena.groups) == 2 and arena.group_of == {0: 0, 1: 1, 2: 1}
    assert arena.states[2] is arena.states[1]
    assert arena.seat.ablatable == [False, True]
    results = arena.play([(0, 3), (2, 0)])
    assert results.sum(1).tolist() == [4, 4]
    rows = arena.seat.slot_rows
    assert (
        arena.seat.ablations.rotary.tolist() == [False] * 2 * rows + [True] * rows + [False] * rows
    )
    assert arena.seat.lane_of == {0: (0, 0), 2: (1, 0), 3: (0, 1)}
    states = []
    for players, matchups in (
        ([relational, yemong, SCRIPTED], [(0, 2), (1, 0)]),
        ([relational, yemong, rotary, SCRIPTED], [(0, 3), (2, 0)]),
    ):
        engine = Arena(players, GAME, CPU, SMALL)
        torch.manual_seed(5)
        engine.play(matchups)
        states.append((_block_state(engine, 0), _block_state(engine, 1)))
    for one, two in zip(states[0][0], states[1][0], strict=True):
        assert torch.equal(one, two)
    assert any(not torch.equal(one, two) for one, two in zip(states[0][1], states[1][1]))


def test_a_block_without_ablations_plays_as_in_an_engine_without_them(tmp_path):
    """Block 0's games are the same in an engine built without ablations as in one whose
    other block plays an ablated copy; the ablated copy's block plays other games."""
    base = _yemong(tmp_path / "a")
    rotary = player(f"{base.name}:rotary")
    states = []
    for players, matchups in (
        ([base, SCRIPTED], [(0, 1), (0, 1)]),
        ([base, SCRIPTED, rotary], [(0, 1), (2, 1)]),
    ):
        # the same lanes in both (the default caps a one-player group at a slot a block):
        # the lanes' draws are one batched draw, shaped by the lane count
        arena = Arena(players, GAME, CPU, SMALL, lanes=(4,))
        torch.manual_seed(5)
        arena.play(matchups)
        states.append((_block_state(arena, 0), _block_state(arena, 1)))
    assert states[0][0]
    for one, two in zip(states[0][0], states[1][0], strict=True):
        assert torch.equal(one, two)
    assert any(not torch.equal(one, two) for one, two in zip(states[0][1], states[1][1]))


# ---- players at a temperature


def test_a_tempered_name_is_its_policy_at_its_temperature(tmp_path):
    from bnb.league.matches import temperature_of

    base = _yemong(tmp_path / "a")
    cold = player(f"{base.name}@T=0.7")
    assert cold == Player(f"{base.name}@T=0.7", policy=base.policy, temperature=0.7)
    ablated = player(f"{base.name}:rotary@T=0.7")
    assert (ablated.ablations, ablated.temperature) == (("rotary",), 0.7)
    assert temperature_of("scripted@T=0.7") == ("scripted@T=0.7", None)  # a teacher's
    assert player("scripted@T=0.7").teacher.temperature == 0.7
    assert temperature_of("a@b") == ("a@b", None)
    for bad, match in (
        ("rung_T2@T=0.5", "only a network"),
        (f"{base.name}@T=0", "positive"),
        (f"{base.name}@T=inf", "positive"),
        (f"{base.name}@T=x", "not a number"),
        (f"{base.name}@T=0.7:rotary", "after any ablations"),  # the temperature comes last
        (f"{base.name}@T=0.5@T=2", "twice"),
    ):
        with pytest.raises(ValueError, match=match):
            player(bad)
    weights = tournament.identity(base)
    assert tournament.identity(cold) == f"{weights}@T=0.7"
    assert tournament.identity(player(f"{base.name}:rotary@T=2")) == f"{weights}:rotary@T=2.0"


def test_temperature_one_samples_exactly_as_without_one():
    """Logits / 1 are the logits, in their own dtype, so the same draws give the same
    sample, log-probability and entropy; a controller at T = 1 decides as without one."""
    from bnb.controllers.relational import RelationalController
    from bnb.game import api

    logits = torch.randn(6, 2, 42).to(torch.bfloat16)
    tempered = policy.tempered(logits, torch.ones(6))
    assert tempered.dtype == torch.bfloat16 and torch.equal(tempered, logits)
    torch.manual_seed(0)
    model = Relational(RelationalBuild(RelationalConfig(hidden=16), 2, 4))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_()
    controller = RelationalController(model)
    rules = api.rules_of(GAME)
    state = api.reset(rules, api.ship_types(), api.uniforms(rules, 3))
    _, obs, _ = api.perceive(rules, state)
    reset = torch.ones(obs.ships.shape[0], dtype=torch.bool)
    decisions = []
    for temperature in (None, torch.ones(obs.ships.shape[0])):
        torch.manual_seed(3)
        action, _, outputs = controller.step(None, obs, reset, temperature=temperature)
        decisions.append((action.packed(), outputs.log_probability, outputs.entropy))
    for one, two in zip(*decisions, strict=True):
        assert torch.equal(one, two)


def test_a_low_temperature_concentrates_and_a_high_one_flattens_only_its_rows(tmp_path):
    """Each row samples at its slot's player's temperature: the mean entropy a block's
    statistics record falls below the untempered policy's at T = 0.25 and rises above it at
    T = 4, the scripted agent's is 0, and a block without a tempered player plays the same
    games as in an engine without one."""
    from bnb.league.matches import STATISTICS

    base = _network(tmp_path / "a", 1)
    cold, hot = player(f"{base.name}@T=0.25"), player(f"{base.name}@T=4")
    arena = Arena([base, cold, hot, SCRIPTED], GAME, CPU, SMALL, lanes=(4,))
    assert arena.states[0] is arena.states[1] is arena.states[2]
    arena.play([(1, 0), (2, 3)])
    rows = arena.seat.slot_rows
    assert arena.seat.network_temperature.tolist() == (
        [0.25] * rows + [1.0] * rows + [4.0] * rows + [1.0] * rows
    )
    statistics = arena.statistics()
    entropy = (
        statistics[..., STATISTICS.index("entropy")]
        / statistics[..., STATISTICS.index("ship_ticks")]
    )
    assert entropy[0, 0] < entropy[0, 1] < entropy[1, 0]
    assert entropy[1, 1] == 0.0  # the scripted agent's
    states = []
    for players, matchups in (
        ([base, SCRIPTED], [(0, 1), (0, 1)]),
        ([base, SCRIPTED, hot], [(0, 1), (2, 1)]),
    ):
        engine = Arena(players, GAME, CPU, SMALL, lanes=(4,))
        torch.manual_seed(5)
        engine.play(matchups)
        states.append((_block_state(engine, 0), _block_state(engine, 1)))
    assert states[0][0]
    for one, two in zip(states[0][0], states[1][0], strict=True):
        assert torch.equal(one, two)
    assert any(not torch.equal(one, two) for one, two in zip(states[0][1], states[1][1]))


def test_the_aspect_statistics_split_each_rows_kills_and_hit_damage():
    """Each row's kills by aspect class sum to its kills (every kill is a hit's), and its
    damage by class is positive where it landed hits."""
    from bnb.contracts import ASPECTS
    from bnb.league.matches import STATISTICS

    game = dataclasses.replace(GAME, arena_radius=600.0, episode_seconds=12.0)
    arena = Arena([SCRIPTED, SCRIPTED], game, CPU, SMALL)
    torch.manual_seed(3)
    arena.play([(0, 1), (1, 0)])
    statistics = arena.statistics()  # (blocks, 2, len(STATISTICS))
    column = {name: statistics[..., number] for number, name in enumerate(STATISTICS)}
    kills = sum(column[f"kills_{aspect}"] for aspect in ASPECTS)
    damage = sum(column[f"damage_{aspect}"] for aspect in ASPECTS)
    assert float(column["hits"].sum()) > 0.0 and float(column["kills"].sum()) > 0.0
    torch.testing.assert_close(kills, column["kills"])
    assert torch.equal(damage > 0.0, column["hits"] > 0.0)


def test_a_deathmatch_block_is_scored_by_its_games_kill_shares():
    """Team deathmatch is rated by kill share (the author, 2026-11-13): each game's score is
    the second player's deaths over both sides' (any cause, as the mode's score counts
    them; 0.5 with none), a block's result (score, 0, games - score); elimination and
    frontline keep win, draw and loss."""
    from bnb.contracts.config import ELIMINATION, FRONTLINE
    from bnb.league.matches import STATISTICS

    game = dataclasses.replace(GAME, arena_radius=600.0, episode_seconds=12.0)
    assert tournament.kill_share_scored(game)
    eliminated = dataclasses.replace(game, mode=ELIMINATION, respawn_seconds=None)
    assert not tournament.kill_share_scored(eliminated)
    assert not tournament.kill_share_scored(dataclasses.replace(game, mode=FRONTLINE))
    arena = Arena([SCRIPTED, RANDOM], game, CPU, SMALL)
    torch.manual_seed(3)
    results = tournament.scored(arena, game, [(0, 1), (1, 0)])
    rows = arena.rollout.carry["controllers"][SEAT].statistics.double()
    deaths = rows[:, STATISTICS.index("deaths")].reshape(2, 2, SMALL.block_environments)
    assert float(deaths.sum()) > 0.0  # the test means something: games with deaths
    for block in range(2):
        shares = [
            0.5 if mine + theirs == 0 else theirs / (mine + theirs)
            for mine, theirs in zip(deaths[block, 0].tolist(), deaths[block, 1].tolist())
        ]
        score = sum(shares)
        expected = [score, 0.0, SMALL.block_environments - score]
        torch.testing.assert_close(results[block], torch.tensor(expected, dtype=torch.float64))
    assert ((results[:, 0] >= 0.0) & (results[:, 0] <= SMALL.block_environments)).all()


def test_a_deathmatch_block_scored_by_win_loss_counts_each_games_kill_lead():
    """With deathmatch_score "win_loss" a team deathmatch game is a win, draw or loss by the
    sign of its final kill difference (the first player's kills are the second's deaths)."""
    from bnb.league.matches import STATISTICS

    game = dataclasses.replace(GAME, arena_radius=600.0, episode_seconds=12.0)
    assert not tournament.kill_share_scored(game, "win_loss")
    assert tournament.scoring_of(game, "win_loss") == "win_loss"
    assert tournament.scoring_of(game) == "kill_share"
    with pytest.raises(ValueError, match="deathmatch score"):
        tournament.kill_share_scored(game, "kill_lead")
    arena = Arena([SCRIPTED, RANDOM], game, CPU, SMALL)
    torch.manual_seed(3)
    results = tournament.scored(arena, game, [(0, 1), (1, 0)], "win_loss")
    rows = arena.rollout.carry["controllers"][SEAT].statistics.double()
    deaths = rows[:, STATISTICS.index("deaths")].reshape(2, 2, SMALL.block_environments)
    lead = torch.sign(deaths[:, 1] - deaths[:, 0])  # (blocks, games): the first player's
    assert bool((lead != 0).any())  # the test means something: decided games
    expected = torch.stack([(lead > 0).sum(-1), (lead == 0).sum(-1), (lead < 0).sum(-1)], -1)
    torch.testing.assert_close(results.double(), expected.double())


def test_a_fractional_summary_reports_the_kill_share_score():
    summary = tournament._summary(torch.tensor([2.75, 0.0, 1.25], dtype=torch.float64))
    assert summary["wins"] == 2.75 and summary["draws"] == 0 and summary["losses"] == 1.25
    assert summary["score"] == pytest.approx(2.75 / 4.0)
    whole = tournament._summary(torch.tensor([3.0, 1.0, 0.0], dtype=torch.float64))
    assert whole["wins"] == 3 and isinstance(whole["wins"], int)


# ---- the allowed pairs (the plan of a batch: test_rating_design.py)


def test_the_scripted_agent_plays_only_its_partners():
    names = ["a", "scripted", "b", "c"]
    allowed = tournament.allowed_pairs(names, ("a", "b"))
    assert allowed[1].tolist() == [True, False, True, False]
    assert allowed[:, 1].tolist() == [True, False, True, False]
    assert bool(allowed[0, 3]) and not bool(allowed.diagonal().any())
    assert int(tournament.allowed_pairs(names).sum()) == 12
    for bad in (("x",), ("scripted",), ()):
        with pytest.raises(ValueError, match="partners"):
            tournament.allowed_pairs(names, bad)


def test_every_player_must_reach_the_anchor():
    allowed = torch.zeros((3, 3), dtype=torch.bool)
    allowed[0, 1] = allowed[1, 0] = True  # player 2 plays no one
    config = tournament.TournamentConfig(SMALL, budget=SMALL.games)
    with pytest.raises(ValueError, match="reach the scripted agent"):
        tournament.fit_field(lambda matchups: None, ["scripted", "a", "b"], 0, allowed, config)


# ---- the tournament


def _bradley_terry(truth: torch.Tensor, block_games: int, generator):
    """A batch's results drawn from the rating model: each block's first player wins each
    game with its probability against the second (no ties)."""

    def play(matchups):
        rows = []
        for first, second in matchups:
            probability = win_probability(truth[first], truth[second])
            games = torch.tensor(float(block_games), dtype=torch.float64)
            won = int(torch.binomial(games, probability, generator))
            rows.append((won, 0, block_games - won))
        return torch.tensor(rows)

    return play


def test_the_tournament_recovers_a_known_field():
    truth = torch.tensor([0.0, 120.0, -200.0, 350.0, 500.0, 250.0], dtype=torch.float64)
    names = ["scripted", "a", "b", "c", "d", "e"]
    allowed = tournament.allowed_pairs(names, ("a", "b"))
    config = tournament.TournamentConfig(
        ArenaConfig(blocks=8, block_environments=32), target_stderr=30.0, budget=40_000
    )
    generator = torch.Generator().manual_seed(3)
    logged = []
    posterior, pair_games = tournament.fit_field(
        _bradley_terry(truth, 32, generator), names, 0, allowed, config, logged.append, "test"
    )
    ratings, stderr = posterior.ratings, posterior.stderr
    assert float(stderr.max()) < 30.0 and len(logged) < 40_000 // 256
    assert float(pair_games[0, 3:].sum()) == 0.0  # the scripted agent played only a and b
    assert ((ratings - truth).abs() <= 4.0 * stderr + 1e-9).all(), (ratings, stderr)


def _kill_shares(truth: torch.Tensor, block_games: int, generator):
    """A batch's kill-share results drawn about the rating model: each game's share a Beta
    draw whose mean is the first player's win probability against the second (a fractional
    outcome with the model's expectation), a block's result (score, 0, games - score)."""

    def play(matchups):
        rows = []
        for first, second in matchups:
            mean = float(win_probability(truth[first], truth[second]))
            beta = torch.distributions.Beta(
                torch.tensor(8.0 * mean, dtype=torch.float64),
                torch.tensor(8.0 * (1.0 - mean), dtype=torch.float64),
            )
            shares = beta.sample((block_games,))
            score = float(shares.sum())
            rows.append((score, 0.0, block_games - score))
        return torch.tensor(rows, dtype=torch.float64)

    return play


def test_the_tournament_recovers_a_known_field_from_kill_shares():
    """The Bradley-Terry fit takes fractional outcomes (a game's kill share) as it takes a
    draw's half: a field whose games are scored by shares with the model's mean is
    recovered within its standard errors, which stay finite and positive."""
    truth = torch.tensor([0.0, 120.0, -200.0, 350.0], dtype=torch.float64)
    names = ["scripted", "a", "b", "c"]
    allowed = tournament.allowed_pairs(names)
    config = tournament.TournamentConfig(
        ArenaConfig(blocks=8, block_environments=32), target_stderr=30.0, budget=40_000
    )
    torch.manual_seed(5)
    generator = torch.Generator().manual_seed(5)
    posterior, _ = tournament.fit_field(
        _kill_shares(truth, 32, generator), names, 0, allowed, config, lambda line: None, "test"
    )
    ratings, stderr = posterior.ratings, posterior.stderr
    assert float(stderr.max()) < 30.0 and bool((stderr[1:] > 0.0).all())
    assert ((ratings - truth).abs() <= 4.0 * stderr + 1e-9).all(), (ratings, stderr)


def test_each_batch_is_recorded_and_a_tournament_resumes_from_its_record(tmp_path):
    truth = torch.tensor([0.0, 120.0, -200.0, 350.0], dtype=torch.float64)
    names = ["scripted", "a", "b", "c"]
    allowed = tournament.allowed_pairs(names, ("a",))
    layout = ArenaConfig(blocks=4, block_environments=16)
    record = tmp_path / "frontline.batches.jsonl"

    def fit(budget, seed):
        config = tournament.TournamentConfig(
            layout, target_stderr=1.0, hopeless=math.inf, budget=budget
        )
        play = _bradley_terry(truth, 16, torch.Generator().manual_seed(seed))
        logged = []
        fitted = tournament.fit_field(play, names, 0, allowed, config, logged.append, "f", record)
        return fitted, logged

    (posterior, pair_games), logged = fit(3 * 64, 1)
    ratings, stderr = posterior.ratings, posterior.stderr
    lines = [json.loads(line) for line in record.read_text().splitlines()]
    assert [line["batch"] for line in lines] == [0, 1, 2] and lines[-1]["games"] == 192
    assert [row["player"] for row in lines[-1]["ratings"]] == names
    assert lines[-1]["ratings"][3]["rating"] == pytest.approx(float(ratings[3]))
    assert lines[-1]["worst_stderr"] == pytest.approx(float(stderr.max()))
    assert all(len(line["matchups"]) == len(line["results"]) == 4 for line in lines)
    assert "at the latest (0 batches)" in logged[-1]
    # a larger budget goes on from the record: its games kept, two batches more
    (_, resumed), logged = fit(5 * 64, 2)
    assert "resumed" in logged[0] and len(record.read_text().splitlines()) == 5
    assert float(resumed.sum()) == 2 * 5 * 64
    assert ((resumed - pair_games) >= 0).all()
    # a finished record plays nothing more
    (_, again), _ = fit(5 * 64, 3)
    assert torch.equal(again, resumed)
    with pytest.raises(ValueError, match="another field"):
        tournament.fit_field(
            lambda matchups: None,
            ["scripted", "a", "b", "d"],
            0,
            allowed,
            tournament.TournamentConfig(layout, budget=64),
            record=record,
        )


# ---- reusing earlier tournaments' games


def _recorded(record, names, identities=None, batches=3, seed=1):
    """A record of `batches` batches of 4 blocks of 16 games over `names` (the scripted agent
    first), its players' identities recorded when given; returns its replayed wins and pair
    games."""
    truth = torch.linspace(0.0, 300.0, len(names), dtype=torch.float64)
    layout = ArenaConfig(blocks=4, block_environments=16)
    config = tournament.TournamentConfig(
        layout, target_stderr=1.0, hopeless=math.inf, budget=batches * 64
    )
    play = _bradley_terry(truth, 16, torch.Generator().manual_seed(seed))
    allowed = tournament.allowed_pairs(names)
    tournament.fit_field(
        play, names, 0, allowed, config, lambda line: None, "f", record, None, identities
    )
    count = len(names)
    wins = torch.zeros((count, count), dtype=torch.float64)
    pair_games = torch.zeros((count, count), dtype=torch.float64)
    tournament._replay(record, names, wins, pair_games)
    return wins, pair_games


def test_a_reused_record_gives_the_shared_players_games_as_its_replay(tmp_path):
    record = tmp_path / "frontline.batches.jsonl"
    old = ["scripted", "a", "b", "c", "d"]
    wins, pair_games = _recorded(
        record, old, [f"sha256:{name}" if name != "scripted" else name for name in old]
    )
    assert json.loads(record.read_text().splitlines()[0])["identities"]["a"] == "sha256:a"
    # a field of some of them, in another order, one under another path, and a newcomer
    names = ["scripted", "moved/c", "a", "e"]
    identities = ["scripted", "sha256:c", "sha256:a", "sha256:e"]
    found = tournament.reuse_record(record, names, identities)
    shared = [0, 3, 1]  # scripted, c, a in the old field
    assert torch.equal(found.wins[:3, :3], wins[shared][:, shared])
    assert torch.equal(found.pair_games[:3, :3], pair_games[shared][:, shared])
    assert float(found.pair_games[3].sum()) == 0.0
    assert found.games == float(found.pair_games.sum()) / 2 > 0
    assert found.matched == {"scripted": "scripted", "a": "a", "c": "moved/c"}
    assert found.batches == 3 and found.recorded == tuple(old) and not found.skipped
    # the same path with other weights is not matched
    other = tournament.reuse_record(record, ["scripted", "a"], ["scripted", "sha256:a2"])
    assert other.matched == {"scripted": "scripted"} and other.games == 0
    # weights two of the field's players share are ambiguous
    twice = tournament.reuse_record(
        record, ["scripted", "a", "a2"], ["scripted", "sha256:a", "sha256:a"]
    )
    assert "a" in twice.skipped and twice.games == 0


def test_a_policy_is_identified_by_its_weights_and_a_scripted_player_by_its_name(tmp_path):
    weights = tournament.identity(_network(tmp_path / "a", 1))
    assert weights.startswith("sha256:")
    moved = player(str((tmp_path / "a").rename(tmp_path / "moved")))
    assert tournament.identity(moved) == weights  # a policy moved keeps its identity
    assert tournament.identity(_network(tmp_path / "a", 2)) != weights  # new weights there
    ablated = dataclasses.replace(moved, name="moved:rotary", ablations=("rotary",))
    assert tournament.identity(ablated) == weights + ":rotary"
    assert tournament.identity(SCRIPTED) == "scripted" and tournament.identity(HALF) == "rung_T2"


def test_a_legacy_record_is_reused_by_its_paths_but_not_a_path_written_after_it(tmp_path):
    a, b = _network(tmp_path / "a", 1), _network(tmp_path / "b", 2)
    record = tmp_path / "frontline.batches.jsonl"
    old = ["scripted", a.name, b.name, "gone"]
    wins, pair_games = _recorded(record, old)  # no identities: a legacy record
    assert "identities" not in json.loads(record.read_text().splitlines()[0])
    field = [SCRIPTED, b, a]
    found = tournament.reuse_record(
        record, [p.name for p in field], [tournament.identity(p) for p in field]
    )
    order = [0, 2, 1]
    assert torch.equal(found.pair_games, pair_games[order][:, order])
    assert torch.equal(found.wins, wins[order][:, order])
    assert set(found.matched) == {"scripted", a.name, b.name} and set(found.skipped) == {"gone"}
    # a rewind: b moved away, and other weights written at its path an hour after the games
    (tmp_path / "rewound_1").mkdir()
    moved = player(str((tmp_path / "b").rename(tmp_path / "rewound_1" / "b")))
    rewritten = _network(tmp_path / "b", 3)
    written = json.loads((tmp_path / "b" / checkpoint.RECORD).read_text())
    later = datetime.datetime.fromisoformat(written["time"]) + datetime.timedelta(hours=1)
    written["time"] = later.isoformat()
    (tmp_path / "b" / checkpoint.RECORD).write_text(json.dumps(written))
    for field in ([SCRIPTED, moved, a], [SCRIPTED, rewritten, a]):
        found = tournament.reuse_record(
            record, [p.name for p in field], [tournament.identity(p) for p in field]
        )
        assert set(found.matched) == {"scripted", a.name} and "written at" in found.skipped[b.name]
        assert float(found.pair_games[1].sum()) == 0.0


def test_a_record_of_another_game_version_is_not_reused(tmp_path):
    """Its games were another game's: nothing of it starts the fit, and every player of it
    is reported skipped with the reason (a record without a version is version 1's)."""
    from bnb.game import VERSION

    a = _network(tmp_path / "a", 1)
    field = [SCRIPTED, a]
    names, identities = [p.name for p in field], [tournament.identity(p) for p in field]
    record = tmp_path / "frontline.batches.jsonl"
    _recorded(record, names, identities)
    assert tournament.reuse_record(record, names, identities).games > 0
    for version in (VERSION - 1, None):
        lines = [json.loads(line) for line in record.read_text().splitlines()]
        for line in lines:
            line.pop("game_version", None)
            if version is not None:
                line["game_version"] = version
        record.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
        found = tournament.reuse_record(record, names, identities)
        assert found.games == 0 and float(found.pair_games.sum()) == 0.0
        assert all("game version" in why for why in found.skipped.values())
        assert set(found.skipped) == set(names)


def test_games_of_one_deathmatch_scoring_are_not_resumed_or_reused_under_the_other(tmp_path):
    """Each batch names its scoring; a record of the other scoring is refused on resume and
    left out of reuse, and a record that names none counts as the default's (kill share)."""
    names = ["scripted", "a", "b"]
    record = tmp_path / "deathmatch.batches.jsonl"
    truth = torch.tensor([0.0, 100.0, 200.0], dtype=torch.float64)
    layout = ArenaConfig(blocks=4, block_environments=16)
    config = tournament.TournamentConfig(layout, target_stderr=1.0, hopeless=math.inf, budget=64)
    play = _bradley_terry(truth, 16, torch.Generator().manual_seed(1))
    allowed = tournament.allowed_pairs(names)
    win_loss, kill_share = ("win_loss", "kill_share"), ("kill_share", "kill_share")
    tournament.fit_field(
        play,
        names,
        0,
        allowed,
        config,
        lambda line: None,
        "d",
        record,
        identities=names,
        scoring=win_loss,
    )
    assert json.loads(record.read_text().splitlines()[0])["scoring"] == "win_loss"
    with pytest.raises(ValueError, match="scored by win_loss, not kill_share"):
        tournament.fit_field(
            play, names, 0, allowed, config, lambda line: None, "d", record, scoring=kill_share
        )
    assert tournament.reuse_record(record, names, names, scoring=win_loss).games > 0
    other = tournament.reuse_record(record, names, names, scoring=kill_share)
    assert other.games == 0 and all("scored by win_loss" in why for why in other.skipped.values())
    # a record without a scoring was scored by the default
    legacy = tmp_path / "legacy.batches.jsonl"
    _recorded(legacy, names, names)
    assert "scoring" not in json.loads(legacy.read_text().splitlines()[0])
    assert tournament.reuse_record(legacy, names, names, scoring=kill_share).games > 0
    assert tournament.reuse_record(legacy, names, names, scoring=win_loss).games == 0
    with pytest.raises(ValueError, match="scored by kill_share, not win_loss"):
        tournament.fit_field(
            play, names, 0, allowed, config, lambda line: None, "d", legacy, scoring=win_loss
        )


def test_a_batchs_full_outcome_scores_as_each_scoring_did_when_it_was_played():
    """`outcomes` keeps a block's wins, draws and losses by the game's win rule, its summed
    kill shares and each side's kills: scored by kill share it is what kill-share scoring
    gave (`Arena.kill_shares`), and scored by win, draw and loss it is the batch's own
    results."""
    from bnb.league.matches import STATISTICS

    game = dataclasses.replace(GAME, arena_radius=600.0, episode_seconds=12.0)
    arena = Arena([SCRIPTED, RANDOM], game, CPU, SMALL)
    torch.manual_seed(3)
    results, extra = tournament.outcomes(arena, [(0, 1), (1, 0)])
    rows = arena.rollout.carry["controllers"][SEAT].statistics.double()
    deaths = rows[:, STATISTICS.index("deaths")].reshape(2, 2, SMALL.block_environments)
    assert float(deaths.sum()) > 0.0  # the test means something: games with deaths
    assert results.dtype == torch.int64 and int(results.sum()) == SMALL.games
    assert extra["kills"] == [[int(deaths[b, 1].sum()), int(deaths[b, 0].sum())] for b in (0, 1)]
    by_share = tournament.score_results(results.tolist(), extra["kill_share"], "kill_share")
    assert torch.equal(torch.tensor(by_share, dtype=torch.float64), arena.kill_shares())
    by_win_loss = tournament.score_results(results.tolist(), extra["kill_share"], "win_loss")
    assert by_win_loss == results.tolist()


def _wins_of(lines: list[dict], names: list[str], scoring: str) -> torch.Tensor:
    """The wins (K, K) a record's lines give under a scoring, draws as half a win."""
    index = {name: number for number, name in enumerate(names)}
    wins = torch.zeros((len(names), len(names)), dtype=torch.float64)
    for line in lines:
        scored = tournament.score_results(line["results"], line["kill_share"], scoring)
        for (first, second), (won, tied, lost) in zip(line["matchups"], scored, strict=True):
            wins[index[first], index[second]] += won + tied / 2
            wins[index[second], index[first]] += lost + tied / 2
    return wins


def test_a_record_keeps_full_outcomes_and_is_refitted_under_either_scoring(tmp_path, monkeypatch):
    """A deathmatch record played under kill share holds each block's wins, draws, losses,
    summed kill shares and kills; a budget of 0 refits it without playing, under kill share
    to the ratings it recorded and under win, draw and loss to the fit of its win counts; it
    is resumed and reused under either scoring."""
    from bnb.league.design import fit_posterior
    from bnb.league.gauge import GAUGE_TOP

    game = dataclasses.replace(GAME, arena_radius=600.0, episode_seconds=12.0)
    record = tmp_path / "deathmatch.batches.jsonl"
    field = [SCRIPTED, RANDOM, HALF]
    names = [p.name for p in field]
    torch.manual_seed(3)
    played = tournament.TournamentConfig(SMALL, target_stderr=1.0, hopeless=math.inf, budget=16)
    tournament.rate(field, "small", CPU, played, lambda line: None, game, record=record)
    lines = [json.loads(line) for line in record.read_text().splitlines()]
    assert len(lines) == 2 and all(line["scoring"] == "kill_share" for line in lines)
    for line in lines:
        assert all(isinstance(count, int) for row in line["results"] for count in row)
        assert len(line["kill_share"]) == len(line["kills"]) == len(line["matchups"])
    # a budget of 0 builds no engine and plays no game
    monkeypatch.setattr(tournament, "Arena", None)
    refit = tournament.TournamentConfig(SMALL, target_stderr=1.0, hopeless=math.inf, budget=0)
    for scoring in tournament.DEATHMATCH_SCORES:
        rows = tournament.rate(
            field,
            "small",
            CPU,
            refit,
            lambda line: None,
            game,
            record=record,
            deathmatch_score=scoring,
        )
        ratings = fit_posterior(_wins_of(lines, names, scoring), 0).ratings
        expected = [float(rating) + GAUGE_TOP for rating in ratings]
        assert [row["calibrated_elo"] for row in rows] == pytest.approx(expected)
        assert sum(row["games"] for row in rows) == 2 * 16
    assert len(record.read_text().splitlines()) == 2  # nothing appended
    recorded = [row["rating"] + GAUGE_TOP for row in lines[-1]["ratings"]]
    kill_share = tournament.rate(field, "small", CPU, refit, lambda line: None, game, record=record)
    assert [row["calibrated_elo"] for row in kill_share] == pytest.approx(recorded)
    # reused under either scoring, into another field (one player left out)
    for scoring in (("win_loss", "kill_share"), ("kill_share", "kill_share")):
        found = tournament.reuse_record(record, ["scripted", "random"], names[:2], scoring=scoring)
        assert found.games > 0 and found.unscored == 0 and not found.skipped
        wins = _wins_of(lines, names, scoring[0])[:2, :2]
        assert torch.equal(found.wins, wins)


def test_a_budget_of_0_with_no_recorded_games_is_refused(tmp_path):
    names = ["scripted", "a"]
    config = tournament.TournamentConfig(ArenaConfig(blocks=4, block_environments=16), 1.0, 0)
    with pytest.raises(ValueError, match="none are recorded"):
        tournament.fit_field(
            lambda matchups: None,
            names,
            0,
            tournament.allowed_pairs(names),
            config,
            record=tmp_path / "none.batches.jsonl",
        )


def test_an_older_record_mixed_with_full_outcomes_is_read_batch_by_batch(tmp_path):
    """A record a live service began in the older format (kill shares only, no scoring
    named) and newer code went on with: under kill share every batch is read; under win,
    draw and loss only the full-outcome batches are reused, and resuming it is refused."""
    from bnb.game import VERSION

    names = ["scripted", "a", "b"]
    full = [
        dict(
            batch=0,
            game_version=VERSION,
            matchups=[["scripted", "a"], ["a", "b"]],
            results=[[3, 1, 0], [0, 0, 4]],
            kill_share=[2.5, 1.0],
            kills=[[6, 2], [1, 5]],
            scoring="win_loss",
            ratings=[dict(player=name, rating=0.0, stderr=1.0) for name in names],
            seconds=1.0,
            wall_time="2026-11-14T00:00:00+00:00",
            identities=dict(zip(names, names, strict=True)),
        )
    ]
    older = dict(full[0], results=[[2.5, 0.0, 1.5], [1.0, 0.0, 3.0]])
    for key in ("kill_share", "kills", "scoring"):
        older.pop(key)
    record = tmp_path / "deathmatch.batches.jsonl"
    record.write_text("".join(json.dumps(line) + "\n" for line in [older, *full]))
    win_loss, kill_share = ("win_loss", "kill_share"), ("kill_share", "kill_share")
    found = tournament.reuse_record(record, names, names, scoring=kill_share)
    assert found.games == 16 and found.unscored == 0
    assert float(found.wins[0, 1]) == 2.5 + 2.5 and float(found.wins[1, 2]) == 1.0 + 1.0
    found = tournament.reuse_record(record, names, names, scoring=win_loss)
    assert found.games == 8 and found.unscored == 1
    assert float(found.wins[0, 1]) == 3.5 and float(found.wins[1, 2]) == 0.0
    count = len(names)
    wins, pair_games = torch.zeros((count, count)), torch.zeros((count, count))
    with pytest.raises(ValueError, match="scored by kill_share, not win_loss"):
        tournament._replay(record, names, wins, pair_games, scoring=win_loss)


def test_reused_games_that_meet_the_target_play_no_batch(tmp_path):
    record = tmp_path / "earlier.batches.jsonl"
    names = ["scripted", "a", "b"]
    _recorded(record, names, names, batches=6)
    found = tournament.reuse_record(record, names, names)
    config = tournament.TournamentConfig(ArenaConfig(blocks=4, block_environments=16), 200.0, 64)

    def play(matchups):
        raise AssertionError("no batch is played")

    own = tmp_path / "own.batches.jsonl"
    posterior, pair_games = tournament.fit_field(
        play,
        names,
        0,
        tournament.allowed_pairs(names),
        config,
        lambda line: None,
        "f",
        own,
        None,
        names,
        (found.wins, found.pair_games),
    )
    assert float(posterior.stderr.max()) < 200.0 and torch.equal(pair_games, found.pair_games)
    assert not own.exists()
    # unmet, the batches go on from the reused games, the budget counting only the new
    logged = []
    tournament.fit_field(
        _bradley_terry(torch.zeros(3, dtype=torch.float64), 16, torch.Generator().manual_seed(4)),
        names,
        0,
        tournament.allowed_pairs(names),
        tournament.TournamentConfig(config.arena, 1.0, 128, hopeless=math.inf),
        logged.append,
        "f",
        own,
        None,
        names,
        (found.wins, found.pair_games),
    )
    lines = [json.loads(line) for line in own.read_text().splitlines()]
    assert [line["games"] for line in lines] == [64, 128]
    assert lines[0]["reused_games"] == found.games == 6 * 64
    assert lines[0]["identities"] == dict(zip(names, names))


def test_the_tournament_reuses_an_earlier_fields_record(tmp_path):
    earlier = tmp_path / "earlier" / "small.batches.jsonl"
    config = tournament.TournamentConfig(SMALL, target_stderr=1.0, hopeless=math.inf, budget=8)
    tournament.rate(
        [RANDOM, SCRIPTED, HALF], "small", CPU, config, lambda line: None, GAME, record=earlier
    )
    line = json.loads(earlier.read_text())
    assert line["identities"] == {"random": "random", "scripted": "scripted", "rung_T2": "rung_T2"}
    # another field: the shared pair's games reused (its own record left out of the reuse),
    # and none played when they meet the target
    own = tmp_path / "own" / "small.batches.jsonl"
    met = tournament.TournamentConfig(SMALL, target_stderr=1e6, budget=8)
    logged = []
    rows = tournament.rate(
        [SCRIPTED, RANDOM], "small", CPU, met, logged.append, GAME, record=own, reuse=(earlier, own)
    )
    shared = sum(
        sum(result)
        for pair, result in zip(line["matchups"], line["results"], strict=True)
        if set(pair) == {"scripted", "random"}
    )
    assert shared > 0 and f"reused {shared} games" in logged[0]
    assert not own.exists() and len(logged) == 1  # no batch played
    assert [row["games"] for row in rows] == [shared, shared]


def test_a_batch_that_outruns_the_game_clock_raises():
    arena = Arena([SCRIPTED, RANDOM], GAME, CPU, SMALL)
    arena.tick_limit = 0  # as if every game outlived its clock
    with pytest.raises(RuntimeError, match="never ends"):
        arena.play([(0, 1), (1, 0)])


def test_the_tournament_pins_the_scripted_agent_at_1000_and_stops_on_its_budget():
    logged = []
    config = tournament.TournamentConfig(SMALL, target_stderr=1.0, hopeless=math.inf, budget=20)
    rows = tournament.rate([RANDOM, SCRIPTED, HALF], "small", CPU, config, logged.append, GAME)
    by_name = {row["player"]: row for row in rows}
    assert by_name["scripted"]["calibrated_elo"] == 1000.0 and by_name["scripted"]["stderr"] == 0.0
    assert by_name["random"]["stderr"] > 0.0  # random is fitted too
    played = sum(row["games"] for row in rows) // 2  # each game counts for both players
    assert played == 16 and len(logged) == 2  # whole batches of 8, within the budget
    assert {row["map"] for row in rows} == {"small"}
    assert all(row["fleet_size"] == 2 and row["mode"] == GAME.mode for row in rows)


def test_the_tournament_needs_its_anchor_and_a_batch_of_budget():
    with pytest.raises(ValueError, match="anchor 'scripted'"):
        tournament.rate([RANDOM, HALF], "small", CPU, game=GAME)
    with pytest.raises(ValueError, match="anchor 'fighter'"):
        tournament.rate([RANDOM, HALF], "small", CPU, game=GAME, anchor="fighter")
    with pytest.raises(ValueError, match="one batch"):
        tournament.TournamentConfig(SMALL, budget=7)


# ---- head-to-head


def test_head_to_head_reports_each_pair_in_total_and_per_preset():
    rows = tournament.head_to_head(
        [SCRIPTED, RANDOM, HALF, RANDOM], 6, ("one", "two"), CPU, SMALL, {"one": GAME, "two": GAME}
    )
    assert [(row["first"], row["second"]) for row in rows] == [
        ("scripted", "random"),
        ("rung_T2", "random"),
    ]
    for row in rows:
        assert row["wins"] + row["draws"] + row["losses"] == 12
        assert sum(part["wins"] for part in row["presets"].values()) == row["wins"]
        assert 0.0 <= row["score"] <= 1.0 and math.isfinite(row["stderr"])


def test_head_to_head_blocks_divide_the_games():
    layout = tournament.head_to_head_layout(3, 100, ArenaConfig(blocks=4, block_environments=32))
    assert (layout.blocks, layout.block_environments) == (4, 20)  # 15 blocks, in 4 batches
    layout = tournament.head_to_head_layout(1, 6, ArenaConfig(blocks=16, block_environments=32))
    assert (layout.blocks, layout.block_environments) == (1, 6)


def test_head_to_head_takes_pairs_and_an_even_game_count():
    with pytest.raises(ValueError, match="pairs"):
        tournament.head_to_head(
            [SCRIPTED, RANDOM, HALF], 4, ("one",), CPU, SMALL, games_of={"one": GAME}
        )
    with pytest.raises(ValueError, match="even"):
        tournament.head_to_head([SCRIPTED, RANDOM], 3, ("one",), CPU, SMALL, games_of={"one": GAME})


def test_the_score_error_is_the_binomial_one_without_draws():
    summary = tournament._summary(torch.tensor([30, 0, 10]))
    assert summary["score"] == 0.75
    assert summary["stderr"] == pytest.approx(math.sqrt(0.75 * 0.25 / 40))


def test_the_commands_parse_their_options():
    import tyro

    from bnb.cli import HeadToHead, Ratings

    pairs = tyro.cli(HeadToHead, args=["a", "b", "scripted", "random", "--games", "8"])
    assert pairs.players == ("a", "b", "scripted", "random") and pairs.games == 8
    rated = tyro.cli(
        Ratings,
        args=["run", "--with", "x", "y", "--target-stderr", "30", "--scripted-plays", "x"]
        + ["--reuse", "r1", "r2"],
    )
    assert rated.run == "run" and rated.with_ == ("x", "y") and rated.target_stderr == 30.0
    assert rated.scripted_plays == ("x",) and rated.league and rated.reuse == ("r1", "r2")


def test_transformed_scripted_names_parse():
    from bnb.league.matches import teacher_of
    from bnb.learn.bc import TeacherConfig

    assert teacher_of("scripted") is None and teacher_of("rung_T2") is None
    assert teacher_of("scripted@T=2,floor=0.05") == TeacherConfig(2.0, 0.05)
    assert teacher_of("scripted@floor=0.01") == TeacherConfig(1.0, 0.01)
    assert teacher_of("scripted@floor=0.01,T=0.5") == TeacherConfig(0.5, 0.01)
    assert teacher_of("scripted@T=4") == TeacherConfig(4.0, 0.0)
    assert player("scripted@T=2").teacher == TeacherConfig(2.0, 0.0)
    for bad in ("scripted@", "scripted@T", "scripted@T=x", "scripted@T=1,T=2", "scripted@x=1"):
        with pytest.raises(ValueError):
            teacher_of(bad)
    with pytest.raises(ValueError):
        teacher_of("scripted@T=0")
    with pytest.raises(ValueError):
        teacher_of("scripted@floor=2")


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_mixed_field_captures_into_a_graph_and_plays_each_pair_as_alone(tmp_path, monkeypatch):
    """On the GPU the lanes' gathers and writes run inside the captured graph, and a block
    of a three-build field plays as the pair alone (greedy networks, as on the CPU)."""
    monkeypatch.setattr(policy, "sample", _greedy)
    cuda = torch.device("cuda")
    first, second = _network(tmp_path / "a", 1), _network(tmp_path / "b", 2, hidden=24)
    third = _network(tmp_path / "c", 3, hidden=32)
    layout = ArenaConfig(blocks=2, block_environments=8, length=8, compiled=False)
    mixed = Arena([first, second, third, SCRIPTED, RANDOM], GAME, cuda, layout)
    alone = Arena([first, second, SCRIPTED, RANDOM], GAME, cuda, layout)
    states = []
    for arena, pairs in ((mixed, [(0, 1), (2, 4)]), (alone, [(0, 1), (2, 3)])):
        torch.manual_seed(5)
        arena.play(pairs)
        assert arena.rollout.graph is not None
        states.append(_block_state(arena, 0))
    for one, two in zip(*states, strict=True):
        assert torch.equal(one, two)


# ---- ship-type players and the balanced field


def test_a_ship_type_player_flies_its_type_on_its_own_team():
    """Each environment's teams fly their players' types, the block's first player on team
    i mod 2; a player without a type flies the game's lineup."""
    from bnb.league.matches import ship_type_players

    fighters, heavies = ship_type_players("standard")
    assert (fighters.ship_type, heavies.ship_type) == (0, 1)
    assert fighters.rung == Rung(1.0)
    arena = Arena([fighters, heavies, SCRIPTED], GAME, CPU, SMALL)
    arena.play([(0, 1), (2, 1)])
    types = arena.rollout.carry["state"].ships.ship_type.tolist()
    assert types[:4] == [[0, 0, 1, 1], [1, 1, 0, 0]] * 2  # block 0: fighters, then heavies
    assert types[4:] == [[0, 1, 1, 1], [1, 1, 0, 1]] * 2  # block 1: the lineup, then heavies


def test_the_arena_keeps_each_blocks_first_game_statistics():
    """Per block and player: kills, deaths, shots, hits, ship-ticks and summed speed, each
    from the first games alone; one side's kills are at most the other's deaths."""
    from bnb.league.matches import STATISTICS

    arena = Arena([SCRIPTED, RANDOM, HALF], GAME, CPU, SMALL)
    arena.play([(0, 1), (2, 1)])
    statistics = arena.statistics()
    assert statistics.shape == (2, 2, len(STATISTICS))
    column = {name: statistics[..., index] for index, name in enumerate(STATISTICS)}
    assert (column["kills"] <= column["deaths"].flip(1)).all()
    assert (column["hits"] <= column["shots"]).all()
    # each block's 4 games of 2 s at 30 Hz, 2 ships a side: at most 4 x 2 x 61 ship-ticks
    assert (column["ship_ticks"] > 0).all() and (column["ship_ticks"] <= 4 * 2 * 61).all()
    assert (column["speed"] > 0).all()
    arena.play([(0, 1), (2, 1)])  # a batch starts its statistics afresh
    assert (arena.statistics()[..., STATISTICS.index("ship_ticks")] <= 4 * 2 * 61).all()


def test_the_balanced_allocation_is_a_round_robin():
    """Each block to the pair with the fewest games: a budget of every pair's games plays
    each pair the same games, and the record keeps a play's extra results."""
    truth = torch.tensor([0.0, 120.0, -200.0, 350.0], dtype=torch.float64)
    names = ["anchor", "a", "b", "c"]
    blocks = tournament.balanced_blocks(torch.zeros(4, 4), 6, 8, ~torch.eye(4, dtype=torch.bool))
    assert sorted(blocks) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    config = tournament.TournamentConfig(
        ArenaConfig(blocks=3, block_environments=8),
        target_stderr=1.0,
        budget=6 * 32,
        allocation="balanced",
    )
    play = _bradley_terry(truth, 8, torch.Generator().manual_seed(4))
    _, pair_games = tournament.fit_field(
        lambda matchups: (play(matchups), dict(extra=len(matchups))),
        names,
        0,
        tournament.allowed_pairs(names),
        config,
        lambda line: None,
    )
    upper = pair_games[torch.triu_indices(4, 4, 1).unbind()]
    assert upper.tolist() == [32.0] * 6
    with pytest.raises(ValueError, match="allocation"):
        tournament.TournamentConfig(allocation="random")


def test_a_reset_takes_each_ships_type_per_environment():
    from bnb.game import api

    rules = api.rules_of(GAME)
    types = api.ship_types()
    lineup = torch.tensor([[1, 1, 0, 0], [0, 1, 0, 1]], dtype=torch.int32)
    state = api.reset(rules, types, api.uniforms(rules, 2), ship_type=lineup)
    assert state.ships.ship_type.tolist() == lineup.tolist()
    assert (
        api.reset(rules, types, api.uniforms(rules, 2)).ships.ship_type.tolist()
        == [[0, 1, 0, 1]] * 2
    )
    with pytest.raises(ValueError, match="ship_type"):
        api.reset(rules, types, api.uniforms(rules, 2), ship_type=lineup[:1])
