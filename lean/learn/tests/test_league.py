"""The league design (author, 2026-11-22, 2026-11-23 and 2026-11-27): half self-play with both
sides training, half league games with only the live side training; the league in two levels
(lean.learn.pool): a pool of members (the yardstick scripted agent, its personas, the two
weight averages, snapshots) with AlphaStar's decaying tallies and PFSP (1 - p)^2 targets with
a uniform floor; level 1 loads members into K slots between updates by their shares
(duplicates allowed, at most one load per free slot, unloaded slots draining their games);
level 2 draws each new game's slot uniformly over the loaded slots with a free row; redraws
only at reset; routing (each network slot once, on its own rows) and the scripted slots'
per-row parameters in the one scripted call; v2's snapshot schedule and retention."""

import dataclasses
import math

import pytest
import torch
from torch import nn

from lean import game, model, scripted
from lean.contract import SEAT_AXIS, split_joint
from lean.learn import League, LearnConfig, WeightAverages, collect, make_optimiser, update
from lean.learn.pool import (
    DRAINING,
    EMA,
    FIRST_SNAPSHOT,
    FREE,
    LOADED,
    RETIRED,
    SCRIPTED,
    UNIFORM,
    NetworkTable,
    SnapshotSchedule,
    retained,
    slot_draws,
)
from lean.learn.rollout import network_actions, scripted_rows, slot_actions
from lean.learn.tests.setting import StepRecorder, add_snapshots, loaded_slots, place, small
from lean.model.agent import seat_view, split_weights, stepper_weights
from lean.scripted.opponent import explain

ENVS = 8  # 4 self-play, 4 league
MILLION = 1_000_000
TWO_PERSONAS = (
    ("cautious", (("caution", -1.5), ("temperature", 0.03))),
    ("bold", (("aggression", 0.8), ("caution", 1.5))),
)


def test_self_play_trains_both_teams_and_league_games_only_the_live_side():
    config, game_config, policy, opponent, state, carry, generator = small(ENVS, league=True)
    rollout = collect(config, game_config, policy, opponent, state, carry, 6, generator)
    own, exists = rollout.observation.own, state.exists
    self_play = ENVS // 2
    assert rollout.seat_environment.tolist() == list(range(ENVS)) + list(range(self_play))
    # Tick 0: every ship is alive. Self-play: the two seats of an environment own disjoint
    # rows that cover every ship of both teams.
    for environment in range(self_play):
        first, second = own[0, environment], own[0, ENVS + environment]
        assert not (first & second).any()
        assert ((first | second) == exists[environment]).all()
        assert first.any() and second.any()
    # League games: one seat, owning the live team only; the opponent's rows train nowhere.
    for environment in range(self_play, ENVS):
        seats = (rollout.seat_environment == environment).nonzero().flatten()
        assert seats.tolist() == [environment]
        owned = own[0, environment]
        assert owned.any() and not (owned == exists[environment]).all()
    # The second seats' rows reach the gradient: a loss on them alone moves the weights.
    rollout.observation.own = own & (torch.arange(own.shape[1]) >= ENVS)[None, :, None]
    before = [parameter.detach().clone() for parameter in policy.parameters()]
    update(config, policy, make_optimiser(config, policy), rollout, generator=generator)
    assert any(
        not torch.equal(old, new) for old, new in zip(before, policy.parameters(), strict=True)
    )


def test_both_sides_of_a_self_play_game_are_the_live_policy(monkeypatch):
    """The game applies the live policy's samples to both teams of a self-play game, and to
    the live team of a league game, whose other team plays the scripted agent's action
    (the yardstick, the only member before the league opens)."""
    recorder = StepRecorder().install(monkeypatch)
    decided = []

    def decide(*arguments):
        action, choice = decide_now(*arguments)
        decided.append(action)
        return action, choice

    decide_now = scripted.decide
    monkeypatch.setattr(scripted, "decide", decide)
    config, game_config, policy, opponent, state, carry, generator = small(ENVS, league=True)
    rollout = collect(config, game_config, policy, opponent, state, carry, 6, generator)
    team = state.team
    for tick, applied in enumerate(recorder.actions):
        for environment in range(ENVS):
            live = team[environment] == rollout.carry.learner_team[environment]  # no game ended
            assert torch.equal(
                applied[environment][live], split_joint(rollout.action[tick, environment])[live]
            )
            if environment < ENVS // 2:
                other = split_joint(rollout.action[tick, ENVS + environment])
            else:
                # The scripted agent's decision on that tick: its second call (the self-play
                # half's teacher first, then the league half, rollout.scripted_moves).
                other = decided[2 * tick + 1][environment - ENVS // 2]
            assert torch.equal(applied[environment][~live], other[~live])


# ---- level 2: the slot draws


def draw_setting(slots=6, seed=0):
    generator = torch.Generator().manual_seed(seed)
    loaded = torch.ones(slots + 1, dtype=torch.bool)
    loaded[-1] = False  # the overflow is never drawn
    return generator, loaded


def test_rows_per_slot_never_exceed_the_capacity_and_only_loaded_slots_take_games():
    """Many ticks of games ending at random: no slot ever holds more rows than its capacity,
    an unloaded (draining) slot takes no new game, and with K C >= L nothing overflows."""
    generator, loaded = draw_setting()
    loaded[4] = False  # draining
    league_envs, slots = 64, 6
    capacity = torch.tensor(math.ceil(1.25 * league_envs / 5))
    slot = torch.full((league_envs,), -1)
    done = torch.ones(league_envs, dtype=torch.bool)
    binding = 0
    for _ in range(400):
        drawn = slot_draws(loaded, capacity, slot, done, generator, slots)
        assert not (done & (drawn == 4)).any() and not (done & (drawn == slots)).any()
        slot = torch.where(done, drawn, slot)
        rows = torch.bincount(slot, minlength=slots + 1)
        assert (rows[:slots] <= capacity).all()
        binding += int((rows == capacity).any())
        done = torch.rand(league_envs, generator=generator) < 0.3
    assert binding > 50  # the capacity was the binding constraint often


def test_level_two_is_uniform_over_the_loaded_slots_with_room():
    generator, loaded = draw_setting()
    loaded[2] = False
    league_envs, slots, rounds = 256, 6, 200
    capacity = torch.tensor(league_envs)  # never binding
    slot = torch.zeros(league_envs, dtype=torch.long)
    done = torch.ones(league_envs, dtype=torch.bool)
    counts = torch.zeros(slots + 1)
    for _ in range(rounds):
        counts += torch.bincount(
            slot_draws(loaded, capacity, slot, done, generator, slots), minlength=slots + 1
        )
    frequency = counts / (league_envs * rounds)
    expected = torch.tensor([0.2, 0.2, 0.0, 0.2, 0.2, 0.2, 0.0])
    sigma = (0.2 * 0.8 / (league_envs * rounds)) ** 0.5
    assert ((frequency - expected).abs() < 5 * sigma).all()


def test_a_full_slots_draws_go_to_the_slots_with_room():
    """Slots 1 and 2 full: every game goes to slot 0 or 3, evenly."""
    generator = torch.Generator().manual_seed(3)
    loaded = torch.tensor([True, True, True, True, False])
    slot = torch.tensor([1] * 400 + [2] * 400 + [0] * 400)
    done = torch.zeros(slot.shape[0], dtype=torch.bool)
    done[800:] = True
    drawn = slot_draws(loaded, torch.tensor(400), slot, done, generator, 4)[800:]
    assert set(drawn.tolist()) <= {0, 3}
    share = (drawn == 0).float().mean()
    assert abs(share - 0.5) < 5 * (0.25 / 400) ** 0.5


def test_without_a_free_row_a_game_goes_to_the_overflow_slot():
    generator = torch.Generator().manual_seed(4)
    loaded = torch.tensor([True, True, False])
    slot = torch.tensor([0, 0, 1, 1, 0, 1])
    done = torch.tensor([False, False, False, False, True, True])
    drawn = slot_draws(loaded, torch.tensor(2), slot, done, generator, 2)
    assert drawn[4:].tolist() == [2, 2]  # both slots full: the overflow (index K + R)


def test_the_slot_draws_compile_fullgraph_so_they_never_read_the_host():
    torch._dynamo.reset()
    generator, loaded = draw_setting()
    slot = torch.arange(32) % 6  # about 3 rows each before the draws
    done = torch.rand(32, generator=generator) < 0.5
    compiled = torch.compile(slot_draws, fullgraph=True, dynamic=True)
    drawn = compiled(loaded, torch.tensor(7), slot, done, None, 6)
    assert (torch.bincount(torch.where(done, drawn, slot), minlength=7)[:6] <= 7).all()


# ---- PFSP weights and tallies


def test_sampling_weights_are_squared_loss_rate_with_a_uniform_floor():
    league = League(LearnConfig())
    floor = LearnConfig().league_floor
    league.present[:4] = True
    # Fixed tallies: win rates 0.2, 0.5 (draws count half), 0.9, and member 3 unplayed (0.5).
    league.games[:4] = torch.tensor([10.0, 4.0, 10.0, 0.0])
    league.wins[:4] = torch.tensor([2.0, 1.0, 9.0, 0.0])
    league.draws[:4] = torch.tensor([0.0, 2.0, 0.0, 0.0])
    assert torch.allclose(league.win_rates()[:4], torch.tensor([0.2, 0.5, 0.9, 0.5]))
    weight = torch.tensor([0.64, 0.25, 0.01, 0.25])
    expected = (1.0 - floor) * weight / weight.sum() + floor * 0.25
    assert torch.allclose(league.probabilities()[:4], expected)
    assert (league.probabilities()[4:] == 0).all()  # absent members are never drawn
    # The floor keeps a member beaten every time in play.
    league.wins[2] = 10.0
    assert league.probabilities()[2] >= floor / 4 - 1e-6


def test_tallies_decay_per_game_and_unplayed_reads_one_half():
    league = League(LearnConfig())
    decay = LearnConfig().league_decay
    assert torch.allclose(league.win_rates(), torch.full((league.members,), 0.5))
    member = torch.tensor([FIRST_SNAPSHOT])
    league.record(member, torch.tensor([True]), torch.tensor([True]), torch.tensor([False]))
    assert torch.isclose(league.games[FIRST_SNAPSHOT], torch.tensor(1.0))
    assert torch.isclose(league.win_rates()[FIRST_SNAPSHOT], torch.tensor(1.0))
    league.record(member, torch.tensor([True]), torch.tensor([False]), torch.tensor([False]))
    assert torch.isclose(league.games[FIRST_SNAPSHOT], torch.tensor(1.0 + decay))
    assert torch.isclose(league.wins[FIRST_SNAPSHOT], torch.tensor(decay))
    assert torch.isclose(league.win_rates()[SCRIPTED], torch.tensor(0.5))  # unplayed
    # Two games ending on one tick equal the two orders averaged; a RETIRED game counts nowhere.
    together = League(LearnConfig())
    together.record(
        torch.tensor([1, 1, 0, RETIRED]),
        torch.tensor([True, True, False, True]),
        torch.tensor([True, False, True, True]),
        torch.tensor([False, False, False, False]),
    )
    assert torch.isclose(together.games[1], torch.tensor(1.0 + decay))
    assert torch.isclose(together.wins[1], torch.tensor((1.0 + decay) / 2))
    assert together.games[0] == 0 and together.games.sum() == together.games[1]


# ---- level 1: loads between updates


class Tiny(nn.Module):
    """A stand-in network for the league's bookkeeping (its weights are copied, never run)."""

    def __init__(self, value=0.0):
        super().__init__()
        self.config = "tiny-league-test"
        self.weight = nn.Parameter(torch.full((3,), value))


def bookkeeping_config(**overrides) -> LearnConfig:
    return dataclasses.replace(LearnConfig(), league_personas=(), **overrides)


def bookkeeping_league(snapshots=3):
    """An open league of the default slots (12 + 4) with `snapshots` snapshots of Tiny
    networks, no games running."""
    config = bookkeeping_config()
    league = League(config)
    policies = [Tiny(float(index)) for index in range(max(1, snapshots))]
    averages = WeightAverages(policies[0], config)
    league.open(policies[0], averages)
    if snapshots:
        add_snapshots(league, policies, averages=averages)
    return league, policies[0], averages


def set_win_rates(league, rates: dict[int, float], games=100.0):
    for member, rate in rates.items():
        league.games[member] = games
        league.wins[member] = rate * games
        league.draws[member] = 0.0


def test_level_one_loads_each_member_by_its_share_and_duplicates_a_large_one():
    """With every slot free (no games), level 1 loads K slots, each member within one slot of
    q K, and a member whose share is several slots holds several (duplicates), each with the
    member's weights (a network) or parameter row (a scripted member)."""
    league, policy, averages = bookkeeping_league(snapshots=3)
    present = [SCRIPTED, EMA, FIRST_SNAPSHOT, FIRST_SNAPSHOT + 1, FIRST_SNAPSHOT + 2]
    set_win_rates(league, {SCRIPTED: 0.05, EMA: 0.5, FIRST_SNAPSHOT: 0.95})
    league.debt = [0.0] * league.members
    for slot in range(league.slot_count):  # no games: every slot can be reloaded
        if league.slot_state[slot] != FREE:
            league._free(slot)
    league.allocate([0] * (league.slot_count + 1), averages, policy)
    held = league.held()
    share = league.probabilities().tolist()
    assert sum(held) == league.loaded_slots == 12
    for member in present:
        assert abs(held[member] - share[member] * 12) < 1.0 + 1e-6, (member, held, share)
    assert held[SCRIPTED] >= 3  # (1 - 0.05)^2 dominates: about a third of the slots
    snapshot = FIRST_SNAPSHOT + 1
    assert len(loaded_slots(league, snapshot)) >= 2
    for slot in loaded_slots(league, snapshot):
        assert torch.equal(
            league.slot_weights[slot]["policy.weight"],
            league.stored[league.snapshot_step[snapshot]]["policy.weight"],
        )
    scripted_slots = loaded_slots(league, SCRIPTED)
    assert not league.slot_network[scripted_slots].any()
    assert (league.slot_persona[scripted_slots] == 0).all()
    assert league.slot_network[loaded_slots(league, FIRST_SNAPSHOT)].all()


def test_level_one_changes_at_most_the_free_slots_and_unloaded_slots_drain():
    """Every loaded slot has games: a large change of the targets loads at most R slots (the
    free ones) in one update, the loaded count stays K, the unloaded slots drain (still their
    member's, no new game reaches them) and free once their last game has ended."""
    league, policy, averages = bookkeeping_league(snapshots=3)
    rows = [5 if state == LOADED else 0 for state in league.slot_state] + [0]
    before = list(league.slot_holder)
    set_win_rates(league, {SCRIPTED: 0.99, EMA: 0.99, FIRST_SNAPSHOT: 0.0})
    loads = league.loads
    league.allocate(rows, averages, policy)
    assert 0 < league.loads - loads <= 4  # R spare slots
    assert sum(league.held()) == 12
    draining = [slot for slot, state in enumerate(league.slot_state) if state == DRAINING]
    assert len(draining) == league.loads - loads
    assert all(league.slot_holder[slot] == before[slot] for slot in draining)
    # Level 2 never deals to a draining slot; a draining slot frees once its games end.
    generator = torch.Generator().manual_seed(0)
    league.capacity.fill_(64)
    slot = torch.tensor([slot for slot, count in enumerate(rows) for _ in range(count)])
    done = torch.zeros_like(slot, dtype=torch.bool)
    done[::3] = True
    _, drawn = league.draw(league.slot_member[slot], slot, done, generator)
    assert not set(drawn[done].tolist()) & set(draining)
    assert torch.equal(drawn[~done], slot[~done])
    league.free_drained([0] * (league.slot_count + 1))
    assert all(league.slot_state[slot] == FREE for slot in draining)


def test_shares_follow_the_targets_over_a_long_simulated_run():
    """Level 1 and level 2 together on the real League (Tiny networks) over 120 updates with
    fixed synthetic win rates and random game lengths (ticks 8 at a time, 512 an update,
    games of 300 to 1,928 ticks, L = 512): past the tallies' first 20 updates, the games dealt
    to each member match its PFSP share within 0.01, and over 10-update windows within 0.05
    total variation; no game changes slot mid-game, no game goes to a draining slot or
    overflows. (The comparison with core's capacity rule:
    docs/rewrite/sections/league-pool/simulate.py.)"""
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    league, policy, averages = bookkeeping_league(snapshots=3)
    rates = torch.tensor([0.2, 0.5, 0.5, 0.3, 0.6, 0.9] + [0.5] * (league.members - 6))
    league_envs = 512
    league.prepare(league_envs, torch.full((league_envs,), -1))
    member, slot = league.draw(
        torch.zeros(league_envs, dtype=torch.long),
        torch.full((league_envs,), -1),
        torch.ones(league_envs, dtype=torch.bool),
        generator,
    )
    remaining = torch.randint(300, 1929, (league_envs,), generator=generator)
    targets, deals = [], []
    for index in range(120):
        targets.append(league.probabilities())
        dealt = league.dealt.clone()
        for step in range(64):
            if step % 16 == 0:
                league.prepare(league_envs, slot)
            remaining = remaining - 8
            done = remaining <= 0
            won = torch.rand(league_envs, generator=generator) < rates[member.clamp(min=0)]
            league.record(member, done, won, torch.zeros_like(done))
            new_member, new_slot = league.draw(member, slot, done, generator)
            assert torch.equal(new_slot[~done], slot[~done])
            assert bool(league.slot_loaded[new_slot[done]].all())
            member, slot = new_member, new_slot
            fresh = torch.randint(300, 1929, (league_envs,), generator=generator)
            remaining = torch.where(done, fresh, remaining)
        deals.append(league.dealt - dealt)
        steps = 10 * MILLION * (index + 1)  # no snapshot falls in (steps, steps + 1]
        member = league.after_update(policy, averages, steps, steps + 1, member, slot)
    target, dealt = torch.stack(targets)[20:], torch.stack(deals)[20:]
    whole = dealt.sum(0) / dealt.sum()
    mean_target = (target * dealt.sum(1, keepdim=True)).sum(0) / dealt.sum()
    assert (whole - mean_target).abs().max() < 0.01, (whole, mean_target)
    for start in range(0, 100, 10):
        games = dealt[start : start + 10]
        realised = games.sum(0) / games.sum()
        aimed = (target[start : start + 10] * games.sum(1, keepdim=True)).sum(0) / games.sum()
        assert 0.5 * (realised - aimed).abs().sum() < 0.05
    assert float(league.overflow_games) == 0.0


def test_opponents_are_redrawn_only_at_reset():
    config, game_config, policy, opponent, state, carry, generator = small(
        16, game_config=game.GameConfig(training_spawn=True, clock_ticks=5), league=True
    )
    league = opponent.league
    member = add_snapshots(
        league, [model.Policy(model.ModelConfig()) for _ in range(3)], carry.member, carry.slot
    )
    carry = dataclasses.replace(carry, member=member)
    changed_at_reset = 0
    for _ in range(12):
        before, slot = carry.member.clone(), carry.slot.clone()
        rollout = collect(config, game_config, policy, opponent, state, carry, 1, generator)
        state, carry = rollout.state, rollout.carry
        done = rollout.done[0, 8:16]  # seats 8..15 are the league environments
        changed = (carry.member != before) | (carry.slot != slot)
        assert not (changed & ~done).any()
        changed_at_reset += int(changed.sum())
    assert changed_at_reset > 0
    playing = carry.member >= 0
    assert (league.slot_member[carry.slot][playing] == carry.member[playing]).all()


# ---- routing


def test_routing_gives_each_environment_its_own_members_step():
    """network_actions runs each network slot once on its own W rows; each league
    environment must get the carry its own member's step gives (as if it ran alone), a member
    held by two slots the same from both, and scripted environments keep their carry and take
    the scripted action."""
    torch.manual_seed(0)
    members = [model.Policy(model.ModelConfig()) for _ in range(3)]
    config, game_config, policy, opponent, state, carry, generator = small(16, league=True)
    league = opponent.league
    averages = WeightAverages(members[-1], config)
    add_snapshots(league, members, averages=averages)  # the EMA's weights the last policy's
    by_member = {FIRST_SNAPSHOT + index: member for index, member in enumerate(members)}
    by_member[EMA] = members[-1]
    chosen = [0, FIRST_SNAPSHOT, FIRST_SNAPSHOT + 1, FIRST_SNAPSHOT + 2, FIRST_SNAPSHOT, EMA, 0]
    carry = place(league, carry, chosen + [FIRST_SNAPSHOT + 2])
    duplicate = loaded_slots(league, FIRST_SNAPSHOT + 2)
    assert len(duplicate) >= 2  # 5 members in 12 slots: each held at least twice
    slot = carry.slot.clone()
    slot[7] = duplicate[1]  # the same member from its second slot
    carry = dataclasses.replace(
        carry,
        slot=slot,
        opponent=torch.randn_like(carry.opponent),
        reset=torch.zeros_like(carry.reset),  # a tick inside the games: the carries count
    )
    league.prepare(8, carry.slot)
    table = league.table
    assert table.positions.shape[0] == league.width == 8  # C = L in the tests' setting
    networks = [
        index
        for index, source in enumerate(league.slot_source)
        if source is not None and source[0] != "scripted"
    ]
    assert [entry[0] for entry in table.slots] == networks
    views = (game.observe(game_config, state, 0), game.observe(game_config, state, 1))
    scripted_action = torch.full((8, state.exists.shape[1], 3), 1)  # the league environments
    action, new_carry = network_actions(config, league, views, carry, scripted_action, generator)
    first = 8
    view = seat_view(views, torch.arange(first, 16), 1 - carry.learner_team[first:])
    for row, member in enumerate(carry.member.tolist()):
        if member == SCRIPTED:
            assert (action[row] == 1).all()
            assert torch.equal(
                new_carry.select(SEAT_AXIS, row), carry.opponent.select(SEAT_AXIS, row)
            )
            continue
        _, _, expected = by_member[member].step(view, carry.opponent, carry.position_offset[first:])
        assert torch.allclose(
            new_carry.select(SEAT_AXIS, row), expected.select(SEAT_AXIS, row), atol=1e-4
        )


def test_skipping_the_empty_slots_changes_no_move():
    """The table runs only the network slots holding weights; the merge fills the others
    with zero fillers. The league side's move must equal the one from running every slot
    (the empty and scripted ones on the live policy's weights): the same actions and carry,
    bitwise, with each slot's sampling seeded by its slot (the random stream is per call)."""
    torch.manual_seed(0)
    config, game_config, policy, opponent, state, carry, generator = small(16, league=True)
    league = opponent.league
    add_snapshots(league, [model.Policy(model.ModelConfig()) for _ in range(2)])
    carry = place(
        league, carry, [0, FIRST_SNAPSHOT, EMA, FIRST_SNAPSHOT + 1, 0, FIRST_SNAPSHOT, 0, EMA]
    )
    carry = dataclasses.replace(
        carry, opponent=torch.randn_like(carry.opponent), reset=torch.zeros_like(carry.reset)
    )
    league.prepare(8, carry.slot)
    skipped = league.table
    assert len(skipped.slots) < skipped.count == league.slot_count
    every = NetworkTable(
        [
            (
                index,
                torch.tensor(index),
                *split_weights(
                    league.template,
                    league.slot_weights[index]
                    if bool(league.slot_network[index])
                    else stepper_weights(policy),
                ),
            )
            for index in range(league.slot_count)
        ],
        skipped.positions,
        skipped.count,
    )
    views = (game.observe(game_config, state, 0), game.observe(game_config, state, 1))
    scripted_action = torch.full((8, state.exists.shape[1], 3), 1)  # the league environments

    def seeded(network_, slot, positions, views, carry, generator):
        generator.manual_seed(int(slot))
        return slot_actions(network_, slot, positions, views, carry, generator)

    moves = []
    for table in (skipped, every):
        league.table = table
        moves.append(
            network_actions(config, league, views, carry, scripted_action, generator, seeded)
        )
    (action, new_carry), (every_action, every_carry) = moves
    assert torch.equal(action, every_action)
    assert torch.equal(new_carry, every_carry)
    assert not torch.equal(new_carry[:, 1], carry.opponent[:, 1])  # the networks ran


# ---- scripted personas: per-row parameters in the one scripted call


def test_persona_slots_play_their_parameters_in_the_one_scripted_call():
    """The league side of a league game in a persona's slot reads that persona's numbers,
    every other ship the yardstick's (the live side: its teacher is the yardstick), in one
    scripted call; each ship's distribution equals the one of a call where every ship has
    that ship's numbers (to float rounding)."""
    torch.manual_seed(0)
    config, game_config, policy, opponent, state, carry, generator = small(
        8, league=True, league_personas=TWO_PERSONAS
    )
    league = opponent.league
    league.open(policy, WeightAverages(policy, config), slot=carry.slot)
    first = league.first_persona
    assert league.labels[first:] == ["cautious", "bold"]
    carry = place(league, carry, [first, SCRIPTED, first + 1, EMA])
    rows = scripted_rows(opponent, state, carry)
    other = state.team != carry.learner_team[:, None]
    league_row = torch.tensor([0, 0, 0, 0, 1, 0, 2, 0])  # each environment's league side's row
    row = torch.where(other, league_row[:, None], 0)
    yardstick = scripted.ScriptedConfig()
    choice = explain(yardstick, state, None, rows)
    for index, (_, named) in enumerate((("yardstick", ()), *TWO_PERSONAS)):
        values = {**yardstick.values, **dict(named)}
        mine = row == index
        for name in rows:
            assert torch.allclose(rows[name][mine], torch.tensor(values[name]))
        everyone = {name: torch.full_like(rows[name], values[name]) for name in rows}
        alone = explain(yardstick, state, None, everyone)
        alive = mine & state.exists & state.alive
        assert alive.any()
        assert torch.allclose(choice.joint[alive], alone.joint[alive], atol=1e-5)


def test_persona_members_run_in_collect():
    config, game_config, policy, opponent, state, carry, generator = small(
        8, league=True, league_personas=TWO_PERSONAS
    )
    league = opponent.league
    league.open(policy, WeightAverages(policy, config), slot=carry.slot)
    carry = place(league, carry, [league.first_persona] * 2 + [league.first_persona + 1] * 2)
    rollout = collect(config, game_config, policy, opponent, state, carry, 3, generator)
    assert torch.equal(rollout.carry.opponent, carry.opponent)  # no network ran there
    assert rollout.action.shape[0] == 3


def test_snapshot_members_run_in_collect_and_update():
    config, game_config, policy, opponent, state, carry, generator = small(6, league=True)
    add_snapshots(opponent.league, [policy], carry.member, carry.slot)
    carry = place(opponent.league, carry, [0, FIRST_SNAPSHOT, FIRST_SNAPSHOT])
    before = carry.opponent.clone()
    rollout = collect(config, game_config, policy, opponent, state, carry, 4, generator)
    after = rollout.carry.opponent
    assert torch.equal(after[:, 0], before[:, 0])  # the scripted member's environment
    assert not torch.equal(after[:, 1:], before[:, 1:])  # the snapshot ran there
    metrics = update(config, policy, make_optimiser(config, policy), rollout, generator=generator)
    assert all(value == value for value in metrics.values())


# ---- a dropped member's games drain


def test_a_dropped_snapshot_finishes_its_running_games_with_its_old_weights():
    """With room for one snapshot, the second replaces the first: the first's running games
    keep their slot and its weights (draining, RETIRED in the tallies) and are not re-dealt;
    the slot frees once those games end."""
    torch.manual_seed(1)
    old, new = model.Policy(model.ModelConfig()), model.Policy(model.ModelConfig())
    config, game_config, policy, opponent, state, carry, generator = small(
        8,
        game_config=game.GameConfig(training_spawn=True, clock_ticks=12),
        league=True,
        snapshot_members=1,
        recent_snapshots=1,
    )
    league = opponent.league
    add_snapshots(league, [old], carry.member, carry.slot)
    carry = place(league, carry, [FIRST_SNAPSHOT, 0, FIRST_SNAPSHOT, 0])
    rollout = collect(config, game_config, policy, opponent, state, carry, 4, generator)
    state, carry = rollout.state, rollout.carry
    old_slot = loaded_slots(league, FIRST_SNAPSHOT)[0]
    playing = carry.slot == old_slot
    assert playing.tolist() == [True, False, True, False]
    member = add_snapshots(league, [new], carry.member, carry.slot)
    carry = dataclasses.replace(carry, member=member)
    assert league.slot_state[old_slot] == DRAINING and league.snapshots == 1
    assert (carry.member[playing] == RETIRED).all() and (carry.slot[playing] == old_slot).all()
    assert old_slot not in loaded_slots(league, FIRST_SNAPSHOT)
    for name, value in old.named_parameters():
        assert torch.equal(league.slot_weights[old_slot][f"policy.{name}"], value)
    # The draining slot still plays: its games' next moves are the old network's.
    league.prepare(4, carry.slot)
    views = (game.observe(game_config, state, 0), game.observe(game_config, state, 1))
    scripted_action = torch.zeros(4, state.exists.shape[1], 3, dtype=torch.long)  # league envs
    _, new_carry = network_actions(config, league, views, carry, scripted_action, generator)
    view = seat_view(views, torch.arange(4, 8), 1 - carry.learner_team[4:])
    _, _, expected = old.step(view, carry.opponent, carry.position_offset[4:])
    for row in (0, 2):
        assert torch.allclose(
            new_carry.select(SEAT_AXIS, row), expected.select(SEAT_AXIS, row), atol=1e-4
        )
    # Not re-dealt: the games run on to the clock and end; then the slot is free.
    ticks = state.tick[4:][playing].clone()
    assert (ticks > 0).all()
    rollout = collect(config, game_config, policy, opponent, state, carry, 4, generator)
    assert not rollout.done[:, 4:8][:, playing].any()
    state, carry = rollout.state, rollout.carry
    assert (state.tick[4:][playing] == ticks + 4).all()
    assert (carry.slot[playing] == old_slot).all() and (carry.member[playing] == RETIRED).all()
    games = league.games.clone()
    running = playing.clone()
    for _ in range(6):
        rollout = collect(config, game_config, policy, opponent, state, carry, 1, generator)
        state, carry = rollout.state, rollout.carry
        running = running & ~rollout.done[0, 4:8]
        assert (carry.slot[running] == old_slot).all()
        if running.any():
            assert league.slot_state[old_slot] == DRAINING
    assert not running.any()  # both games ended within the clock
    assert (carry.slot != old_slot).all()
    league.prepare(4, carry.slot)
    assert league.slot_state[old_slot] == FREE
    assert league.games.sum() >= games.sum()  # only the other members' games were tallied
    assert float(league.overflow_games) == 0.0


# ---- members: the averages


def test_the_averages_join_at_rl_start_and_follow_their_weights():
    """The EMA joins when the league opens (RL's start, before the first snapshot), the
    uniform average once it holds an update; their slots take their new weights after every
    update."""
    config = bookkeeping_config(warmup_steps=0)
    policy = model.Policy(model.ModelConfig())
    averages = WeightAverages(policy, config)
    league = League(config)
    assert league.present.tolist()[:4] == [True, False, False, False]  # BC: the yardstick
    league.open(policy, averages)
    assert league.present[[SCRIPTED, EMA]].all() and not league.present[UNIFORM]
    assert loaded_slots(league, EMA)  # playing from RL's first tick
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.add_(1.0)
    averages.update(policy, adam_steps=1)  # the uniform average now holds an update
    empty = torch.zeros(0, dtype=torch.long)
    league.after_update(policy, averages, 0, 1, empty, empty)
    assert league.present[[SCRIPTED, EMA, UNIFORM]].all() and league.snapshots == 0
    name, _ = next(iter(policy.named_parameters()))
    for member, which in ((EMA, "ema"), (UNIFORM, "uniform")):
        assert loaded_slots(league, member)
        for slot in loaded_slots(league, member):
            assert torch.equal(
                league.slot_weights[slot][f"policy.{name}"], getattr(averages, which)[name]
            )
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.add_(1.0)
    averages.update(policy, adam_steps=2)
    league.after_update(policy, averages, 1, 2, empty, empty)
    for slot in loaded_slots(league, EMA):
        assert torch.equal(league.slot_weights[slot][f"policy.{name}"], averages.ema[name])


# ---- checkpoints


def test_a_league_state_round_trips():
    """state() then load(): the same slots, debts, tallies, snapshots and slot weights."""
    league, policy, averages = bookkeeping_league(snapshots=3)
    set_win_rates(league, {SCRIPTED: 0.3, FIRST_SNAPSHOT: 0.7})
    rows = [2 if state == LOADED else 0 for state in league.slot_state] + [0]
    league.allocate(rows, averages, policy)
    restored = League(bookkeeping_config())
    restored.load(league.state(), policy, averages)
    for name in League.TENSORS:
        assert torch.equal(getattr(league, name), getattr(restored, name)), name
    assert restored.slot_state == league.slot_state and restored.slot_source == league.slot_source
    assert restored.debt == league.debt and restored.stored.keys() == league.stored.keys()
    for slot, weights in enumerate(league.slot_weights):
        if league.slot_state[slot] != FREE and bool(league.slot_network[slot]):
            for name, value in weights.items():
                assert torch.equal(restored.slot_weights[slot][name], value)


# ---- the schedule and retention


def test_snapshot_steps_are_v2s_league_schedule():
    """v2's LEAGUE_SNAPSHOTS (src/bnb/runtime/snapshots.py): 5, 10, 15, 20, 25, 30, 40, 50, 60,
    75, 90, 110, 130 and 160M, then the last gap (30M)."""
    schedule = SnapshotSchedule(
        LearnConfig().league_snapshot_steps, LearnConfig().league_snapshot_period
    )
    expected = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 110, 130, 160, 190, 220, 250]
    assert schedule.points(250 * MILLION) == [step * MILLION for step in expected]
    # A league run at parity (1,048,576 steps an update) snapshots at the first update past each.
    league = League(LearnConfig())
    policy = Tiny()
    averages = WeightAverages(policy, LearnConfig())
    league.open(policy, averages)
    parity = LearnConfig()
    per_update = parity.environments * parity.segments_per_update * parity.segment_ticks
    taken_at = []
    empty = torch.zeros(0, dtype=torch.long)
    for update_index in range(240):
        before, after = update_index * per_update, (update_index + 1) * per_update
        league.after_update(policy, averages, before, after, empty, empty)
        if league.taken > len(taken_at):
            taken_at.append(after)
    assert taken_at == [math.ceil(step * MILLION / per_update) * per_update for step in expected]
    assert league.snapshots == 10  # retention from the 11th (75M)
    assert len(league.stored) == 10  # the dropped snapshots' host weights are let go


@pytest.mark.parametrize(
    ("taken", "ages"),
    [
        (10, list(range(10))),
        (11, list(range(9)) + [10]),
        (26, list(range(9)) + [10, 25]),
        (40, list(range(10)) + [39]),
    ],
)
def test_retention_is_v2s(taken, ages):
    """v2's bnb.league.pool.retained applied after each snapshot (ages from the newest; kept
    sets recorded from v2's function on 2026-11-23)."""
    kept = set()
    for ordinal in range(taken):
        kept = retained(kept | {ordinal}, 8, 3, 12)
    assert sorted(taken - 1 - ordinal for ordinal in kept) == ages


def test_retention_keeps_at_most_the_cap_and_drops_the_oldest_past_it():
    kept = set()
    for ordinal in range(400):
        kept = retained(kept | {ordinal}, 8, 3, 12)
        assert len(kept) <= 12
    capped = retained(set(range(40)), 8, 3, 4)
    assert capped == set(range(36, 40))
