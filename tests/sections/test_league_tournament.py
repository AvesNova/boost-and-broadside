"""Published ratings and head-to-head (league spec 6): the match engine on fixed slots, the
allocation of its blocks, the allowed pairs, the tournament's anchor, stopping rule and
recovery of a known field, and head-to-head's report, on a small 2v2 game on the CPU."""

import json
import math

import pytest
import torch
from torch.utils import _pytree

from bnb.contracts import GameConfig
from bnb.league import tournament
from bnb.league.matches import SEAT, Arena, ArenaConfig, Player, default_lanes, player
from bnb.league.ratings import SCALE, _fisher, allocate_blocks, win_probability
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
    assert default_lanes([None, 0, None], 16) == (16,)
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
    config = tournament.TournamentConfig(SMALL, target_stderr=1.0, budget=16)
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
        arena = Arena(players, GAME, CPU, SMALL)
        torch.manual_seed(5)
        arena.play(matchups)
        states.append((_block_state(arena, 0), _block_state(arena, 1)))
    assert states[0][0]
    for one, two in zip(states[0][0], states[1][0], strict=True):
        assert torch.equal(one, two)
    assert any(not torch.equal(one, two) for one, two in zip(states[0][1], states[1][1]))


# ---- the allocation of blocks and the allowed pairs


def _trace_after(pair_games, ratings, anchor, pair, games):
    """The summed rating variance with `games` more between `pair`, by direct inversion."""
    count = pair_games.shape[0]
    added = pair_games.clone()
    added[pair[0], pair[1]] += games
    added[pair[1], pair[0]] += games
    free = torch.arange(count) != anchor
    added[free, anchor] += tournament.PRIOR_GAMES
    added[anchor, free] += tournament.PRIOR_GAMES
    return float(torch.linalg.inv(_fisher(ratings / SCALE, added, free)).trace())


def test_each_block_goes_to_the_pair_that_most_reduces_the_variance():
    pair_games = torch.tensor(
        [[0, 30, 4, 0], [30, 0, 10, 2], [4, 10, 0, 0], [0, 2, 0, 0]], dtype=torch.float64
    )
    ratings = torch.tensor([0.0, 150.0, -80.0, 300.0], dtype=torch.float64)
    allowed = tournament.allowed_pairs(["scripted", "a", "b", "c"], ("a", "b"))
    chosen = allocate_blocks(pair_games, ratings, 0, 2, 8, allowed, tournament.PRIOR_GAMES)
    candidates = [(i, j) for i in range(4) for j in range(i + 1, 4) if allowed[i, j]]
    best = min(candidates, key=lambda pair: _trace_after(pair_games, ratings, 0, pair, 8))
    assert chosen[0] == best
    # the second block's choice accounts for the first's games
    after = pair_games.clone()
    after[best[0], best[1]] += 8
    after[best[1], best[0]] += 8
    second = min(candidates, key=lambda pair: _trace_after(after, ratings, 0, pair, 8))
    assert chosen[1] == second


def test_the_blocks_spread_over_the_field_and_keep_to_the_allowed_pairs():
    names = ["scripted", "a", "b", "c", "d", "e"]
    allowed = tournament.allowed_pairs(names, ("a",))
    chosen = allocate_blocks(torch.zeros((6, 6)), torch.zeros(6), 0, 12, 8, allowed, 1.0)
    assert all(allowed[i, j] and i < j for i, j in chosen)
    assert {player for pair in chosen for player in pair} == set(range(6))
    # a pair with many games already is chosen less than one with none
    pair_games = torch.zeros((6, 6))
    pair_games[2, 3] = pair_games[3, 2] = 400
    chosen = allocate_blocks(pair_games, torch.zeros(6), 0, 12, 8, allowed, 1.0)
    assert chosen.count((2, 3)) < chosen.count((4, 5))


def test_the_blocks_keep_to_the_engines_lanes():
    """A batch holds at most a group's lanes of its slots: without them the allocation would
    put more of group 0 (players 1 and 2) in it."""
    names = ["scripted", "a", "b", "c"]
    allowed = tournament.allowed_pairs(names)
    free = allocate_blocks(torch.zeros((4, 4)), torch.zeros(4), 0, 8, 8, allowed, 1.0)
    groups, lanes = [None, 0, 0, 1], (3, 13)
    assert sum(1 for pair in free for index in pair if groups[index] == 0) > 3
    chosen = allocate_blocks(
        torch.zeros((4, 4)), torch.zeros(4), 0, 8, 8, allowed, 1.0, groups, lanes
    )
    held = [groups[index] for pair in chosen for index in pair]
    assert held.count(0) <= 3 and held.count(1) <= 13
    with pytest.raises(ValueError, match="no allowed pair fits"):
        allocate_blocks(
            torch.zeros((3, 3)),
            torch.zeros(3),
            0,
            2,
            8,
            ~torch.eye(3, dtype=torch.bool),
            1.0,
            [None, 0, 0],
            (1,),
        )


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
    ratings, stderr, pair_games = tournament.fit_field(
        _bradley_terry(truth, 32, generator), names, 0, allowed, config, logged.append, "test"
    )
    assert float(stderr.max()) < 30.0 and len(logged) < 40_000 // 256
    assert float(pair_games[0, 3:].sum()) == 0.0  # the scripted agent played only a and b
    assert ((ratings - truth).abs() <= 4.0 * stderr + 1e-9).all(), (ratings, stderr)


def test_each_batch_is_recorded_and_a_tournament_resumes_from_its_record(tmp_path):
    truth = torch.tensor([0.0, 120.0, -200.0, 350.0], dtype=torch.float64)
    names = ["scripted", "a", "b", "c"]
    allowed = tournament.allowed_pairs(names, ("a",))
    layout = ArenaConfig(blocks=4, block_environments=16)
    record = tmp_path / "frontline.batches.jsonl"

    def fit(budget, seed):
        config = tournament.TournamentConfig(layout, target_stderr=1.0, budget=budget)
        play = _bradley_terry(truth, 16, torch.Generator().manual_seed(seed))
        logged = []
        fitted = tournament.fit_field(play, names, 0, allowed, config, logged.append, "f", record)
        return fitted, logged

    (ratings, stderr, pair_games), logged = fit(3 * 64, 1)
    lines = [json.loads(line) for line in record.read_text().splitlines()]
    assert [line["batch"] for line in lines] == [0, 1, 2] and lines[-1]["games"] == 192
    assert [row["player"] for row in lines[-1]["ratings"]] == names
    assert lines[-1]["ratings"][3]["rating"] == pytest.approx(float(ratings[3]))
    assert lines[-1]["worst_stderr"] == pytest.approx(float(stderr.max()))
    assert all(len(line["matchups"]) == len(line["results"]) == 4 for line in lines)
    assert "at the latest (0 batches)" in logged[-1]
    # a larger budget goes on from the record: its games kept, two batches more
    (_, _, resumed), logged = fit(5 * 64, 2)
    assert "resumed" in logged[0] and len(record.read_text().splitlines()) == 5
    assert float(resumed.sum()) == 2 * 5 * 64
    assert ((resumed - pair_games) >= 0).all()
    # a finished record plays nothing more
    (_, _, again), _ = fit(5 * 64, 3)
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


def test_a_batch_that_outruns_the_game_clock_raises():
    arena = Arena([SCRIPTED, RANDOM], GAME, CPU, SMALL)
    arena.tick_limit = 0  # as if every game outlived its clock
    with pytest.raises(RuntimeError, match="never ends"):
        arena.play([(0, 1), (1, 0)])


def test_the_tournament_pins_the_scripted_agent_at_1000_and_stops_on_its_budget():
    logged = []
    config = tournament.TournamentConfig(SMALL, target_stderr=1.0, budget=20)
    rows = tournament.rate([RANDOM, SCRIPTED, HALF], "small", CPU, config, logged.append, GAME)
    by_name = {row["player"]: row for row in rows}
    assert by_name["scripted"]["calibrated_elo"] == 1000.0 and by_name["scripted"]["stderr"] == 0.0
    assert by_name["random"]["stderr"] > 0.0  # random is fitted too
    played = sum(row["games"] for row in rows) // 2  # each game counts for both players
    assert played == 16 and len(logged) == 2  # whole batches of 8, within the budget
    assert {row["map"] for row in rows} == {"small"}
    assert all(row["fleet_size"] == 2 and row["mode"] == GAME.mode for row in rows)


def test_the_tournament_needs_the_scripted_agent_and_a_batch_of_budget():
    with pytest.raises(ValueError, match="scripted agent"):
        tournament.rate([RANDOM, HALF], "small", CPU, game=GAME)
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
        args=["run", "--with", "x", "y", "--target-stderr", "30", "--scripted-plays", "x"],
    )
    assert rated.run == "run" and rated.with_ == ("x", "y") and rated.target_stderr == 30.0
    assert rated.scripted_plays == ("x",) and rated.league


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
