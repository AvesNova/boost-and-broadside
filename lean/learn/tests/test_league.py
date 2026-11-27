"""The league design (author, 2026-11-22 and 2026-11-23): half self-play with both sides
training, half league games with only the live side training; the scripted agent, the two
weight averages and snapshots are members; PFSP (1 - p)^2 with a uniform floor within
row capacities of a factor of even shares; AlphaStar's decaying tallies; redraws only at reset;
capacity routing (each slot's network once, on its own rows); a dropped snapshot finishes
its games in a retiring slot; v2's snapshot schedule and retention."""

import dataclasses
import math

import pytest
import torch

from lean import game, model, scripted
from lean.contract import SEAT_AXIS, split_joint
from lean.learn import League, LearnConfig, WeightAverages, abandon, collect, make_optimiser, update
from lean.learn.pool import (
    EMA,
    FIRST_SNAPSHOT,
    RETIRED,
    SCRIPTED,
    UNIFORM,
    SnapshotSchedule,
    capacity_draws,
    pfsp,
    retained,
)
from lean.learn.rollout import network_actions
from lean.learn.tests.setting import StepRecorder, add_snapshots, small
from lean.model.agent import seat_view

ENVS = 8  # 4 self-play, 4 league
MILLION = 1_000_000


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
    (the scripted member, the only one before a snapshot)."""
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
                other = decided[tick][environment]  # the scripted agent's decision on that tick
            assert torch.equal(applied[environment][~live], other[~live])


# ---- capacity draws


def draws_setting(members=6, seed=0):
    generator = torch.Generator().manual_seed(seed)
    win_rate = torch.tensor([0.1, 0.2, 0.9, 0.5, 0.3, 0.6])[:members]
    present = torch.ones(members, dtype=torch.bool)
    return generator, pfsp(win_rate, present, 0.125), present


def test_rows_per_member_never_exceed_the_capacity():
    """A strongly skewed PFSP (member 0 has 34% of the weight against a capacity of 1.25 / 6 of
    the rows) over many ticks of games ending at random: no member ever holds more rows than
    its capacity, and every ended game gets a member that is present."""
    generator, probability, present = draws_setting()
    present[4] = False  # an absent member is never drawn
    probability = pfsp(torch.tensor([0.1, 0.2, 0.9, 0.5, 0.3, 0.6]), present, 0.125)
    league_envs, members = 64, 6
    capacity = torch.tensor(math.ceil(1.25 * league_envs / int(present.sum())))
    member = torch.zeros(league_envs, dtype=torch.long)
    done = torch.ones(league_envs, dtype=torch.bool)
    binding = 0
    for _ in range(400):
        drawn = capacity_draws(probability, present, capacity, member, done, generator, members)
        member = torch.where(done, drawn, member)
        rows = torch.bincount(member, minlength=members)
        assert (rows <= capacity).all() and rows[4] == 0
        binding += int((rows == capacity).any())
        done = torch.rand(league_envs, generator=generator) < 0.3
    assert binding > 100  # the capacity was the binding constraint often


def test_draws_follow_the_pfsp_weights_when_no_member_is_full():
    generator, probability, present = draws_setting()
    league_envs, members, rounds = 256, 6, 200
    capacity = torch.tensor(league_envs)  # never binding
    member = torch.zeros(league_envs, dtype=torch.long)
    done = torch.ones(league_envs, dtype=torch.bool)
    counts = torch.zeros(members)
    for _ in range(rounds):
        drawn = capacity_draws(probability, present, capacity, member, done, generator, members)
        counts += torch.bincount(drawn, minlength=members)
    samples = league_envs * rounds
    frequency = counts / samples
    sigma = (probability * (1 - probability) / samples).sqrt()
    assert ((frequency - probability).abs() < 5 * sigma).all()


def test_a_full_members_draws_go_to_the_members_with_room():
    """Members 1 and 2 full: every draw takes member 0 or 3, in PFSP proportion between them
    (neither of them fills)."""
    generator = torch.Generator().manual_seed(3)
    present = torch.ones(4, dtype=torch.bool)
    probability = pfsp(torch.tensor([0.5, 0.1, 0.1, 0.8]), present, 0.125)
    member = torch.tensor([1] * 400 + [2] * 400 + [0] * 400)
    done = torch.zeros(member.shape[0], dtype=torch.bool)
    done[800:] = True
    drawn = capacity_draws(probability, present, torch.tensor(400), member, done, generator, 4)[
        800:
    ]
    assert set(drawn.tolist()) <= {0, 3}
    share = (drawn == 0).float().mean()
    expected = probability[0] / (probability[0] + probability[3])
    assert abs(share - expected) < 5 * (expected * (1 - expected) / 400).sqrt()


def test_the_draws_compile_fullgraph_so_they_never_read_the_host():
    torch._dynamo.reset()
    generator, probability, present = draws_setting()
    member = torch.zeros(32, dtype=torch.long)
    done = torch.rand(32, generator=generator) < 0.5
    compiled = torch.compile(capacity_draws, fullgraph=True, dynamic=True)
    drawn = compiled(probability, present, torch.tensor(7), member, done, None, 6)
    assert (torch.bincount(torch.where(done, drawn, member), minlength=6)[1:] <= 7).all()


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


def test_opponents_are_redrawn_only_at_reset():
    config, game_config, policy, opponent, state, carry, generator = small(
        16, game_config=game.GameConfig(training_spawn=True, clock_ticks=5), league=True
    )
    add_snapshots(opponent.league, [model.Policy(model.ModelConfig()) for _ in range(3)])
    changed_at_reset = 0
    for _ in range(12):
        before = carry.member.clone()
        rollout = collect(config, game_config, policy, opponent, state, carry, 1, generator)
        state, carry = rollout.state, rollout.carry
        done = rollout.done[0, 8:16]  # seats 8..15 are the league environments
        changed = carry.member != before
        assert not (changed & ~done).any()
        changed_at_reset += int(changed.sum())
    assert changed_at_reset > 0
    assert (carry.slot == opponent.league.member_slot[carry.member]).all()


# ---- routing


def test_routing_gives_each_environment_its_own_members_step():
    """network_actions runs each slot's network once on its own W rows; each
    league environment must get the carry its own member's step gives (as if it ran alone),
    and scripted environments keep theirs and take the scripted action."""
    torch.manual_seed(0)
    members = [model.Policy(model.ModelConfig()) for _ in range(3)]
    config, game_config, policy, opponent, state, carry, generator = small(16, league=True)
    league = opponent.league
    add_snapshots(league, members)  # the EMA joins with the first, its weights the last policy's
    by_member = {FIRST_SNAPSHOT + index: member for index, member in enumerate(members)}
    by_member[EMA] = members[-1]
    chosen = torch.tensor(
        [
            0,
            FIRST_SNAPSHOT,
            FIRST_SNAPSHOT + 1,
            FIRST_SNAPSHOT + 2,
            FIRST_SNAPSHOT,
            EMA,
            0,
            FIRST_SNAPSHOT + 2,
        ]
    )
    carry = dataclasses.replace(
        carry,
        member=chosen,
        slot=league.member_slot[chosen],
        opponent=torch.randn_like(carry.opponent),
        reset=torch.zeros_like(carry.reset),  # a tick inside the games: the carries count
    )
    league.prepare(8, carry.slot)
    table = league.table
    assert table.positions.shape[0] == league.width == 8  # C = L in the tests' setting
    assert len(table.slots) == league.slot_count  # every slot, the empty ones on no rows
    views = (game.observe(game_config, state, 0), game.observe(game_config, state, 1))
    scripted_action = torch.full((16, state.exists.shape[1], 3), 1)
    action, new_carry = network_actions(config, league, views, carry, scripted_action, generator)
    first = 8
    view = seat_view(views, torch.arange(first, 16), 1 - carry.learner_team[first:])
    for row, member in enumerate(chosen.tolist()):
        if member == SCRIPTED:
            assert (action[row] == 1).all()
            assert torch.equal(
                new_carry.select(SEAT_AXIS, row), carry.opponent.select(SEAT_AXIS, row)
            )
            continue
        _, _, expected = by_member[member].step(view, carry.opponent, carry.position_offset[first:])
        # vmap runs the math attention kernel, the plain step the fused one: float32 rounding apart.
        assert torch.allclose(
            new_carry.select(SEAT_AXIS, row), expected.select(SEAT_AXIS, row), atol=1e-4
        )


def test_snapshot_members_run_in_collect_and_update():
    config, game_config, policy, opponent, state, carry, generator = small(6, league=True)
    add_snapshots(opponent.league, [policy])
    chosen = torch.tensor([0, FIRST_SNAPSHOT, FIRST_SNAPSHOT])  # environments 3..5 are league games
    carry = dataclasses.replace(carry, member=chosen, slot=opponent.league.member_slot[chosen])
    before = carry.opponent.clone()
    rollout = collect(config, game_config, policy, opponent, state, carry, 4, generator)
    after = rollout.carry.opponent
    assert torch.equal(after[:, 0], before[:, 0])  # the scripted member's environment
    assert not torch.equal(after[:, 1:], before[:, 1:])  # the snapshot ran there
    metrics = update(config, policy, make_optimiser(config, policy), rollout, generator=generator)
    assert all(value == value for value in metrics.values())


# ---- retiring slots


def test_a_dropped_snapshot_finishes_its_running_games_with_its_old_weights():
    """With room for one snapshot, the second replaces the first: the first's running game
    keeps its slot and weights (retiring, RETIRED in the tallies) and is not re-dealt; the
    slot frees once that game ends."""
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
    add_snapshots(league, [old])
    chosen = torch.tensor([FIRST_SNAPSHOT, 0, FIRST_SNAPSHOT, 0])
    carry = dataclasses.replace(carry, member=chosen, slot=league.member_slot[chosen])
    rollout = collect(config, game_config, policy, opponent, state, carry, 4, generator)
    state, carry = rollout.state, rollout.carry
    old_slot = int(league.member_slot[FIRST_SNAPSHOT])
    playing = carry.slot == old_slot
    assert playing.tolist() == [True, False, True, False]
    member = add_snapshots(league, [new], carry.member, carry.slot)
    carry = dataclasses.replace(carry, member=member)
    assert league.retiring == {old_slot} and league.snapshots == 1
    assert (carry.member[playing] == RETIRED).all() and (carry.slot[playing] == old_slot).all()
    new_slot = int(league.member_slot[FIRST_SNAPSHOT])
    assert new_slot != old_slot
    for name, value in old.named_parameters():
        assert torch.equal(league.slot_weights[old_slot][f"policy.{name}"], value)
    # The retiring slot still plays: its games' next moves are the old network's.
    league.prepare(4, carry.slot)
    views = (game.observe(game_config, state, 0), game.observe(game_config, state, 1))
    scripted_action = torch.zeros(8, state.exists.shape[1], 3, dtype=torch.long)
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
    # Review B1: after collected ticks the games still sit in the retiring slot (the scripted
    # agent did not take over) and are still RETIRED.
    assert (carry.slot[playing] == old_slot).all() and (carry.member[playing] == RETIRED).all()
    games = league.games.clone()
    # Tick by tick to the end: until its game ends, each keeps the old slot, and the old
    # slot keeps its weights (it is not freed while a game plays in it).
    running = playing.clone()
    for _ in range(6):
        rollout = collect(config, game_config, policy, opponent, state, carry, 1, generator)
        state, carry = rollout.state, rollout.carry
        running = running & ~rollout.done[0, 4:8]
        assert (carry.slot[running] == old_slot).all()
        if running.any():
            assert old_slot in league.retiring and league.slot_weights[old_slot] is not None
    assert not running.any()  # both games ended within the clock
    assert (carry.slot != old_slot).all()
    league.prepare(4, carry.slot)
    assert league.retiring == set() and league.slot_weights[old_slot] is None
    assert league.games.sum() >= games.sum()  # only the other members' games were tallied
    assert league.abandoned == 0


def test_without_a_free_slot_the_retiring_games_are_re_dealt():
    torch.manual_seed(2)
    config, game_config, policy, opponent, state, carry, generator = small(
        8,
        league=True,
        snapshot_members=1,
        recent_snapshots=1,
        retiring_slots=0,
    )
    league = opponent.league
    add_snapshots(league, [policy])
    chosen = torch.full((4,), FIRST_SNAPSHOT)
    carry = dataclasses.replace(carry, member=chosen, slot=league.member_slot[chosen])
    step = league.schedule.steps[1]
    member, lost = league.after_update(
        policy, WeightAverages(policy, config), step - 1, step, carry.member, carry.slot
    )
    assert lost is not None and league.abandoned == 4
    carry = dataclasses.replace(carry, member=member)
    state, carry = abandon(config, game_config, policy, opponent, state, carry, lost, generator)
    assert carry.reset[4:].all() and (carry.member != RETIRED).all()


# ---- members: the averages


def test_the_averages_join_with_the_first_snapshot_and_follow_their_weights():
    config = LearnConfig(warmup_steps=0)
    policy = model.Policy(model.ModelConfig())
    averages = WeightAverages(policy, config)
    league = League(config)
    league.after_update(
        policy, averages, 0, 1, torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
    )
    assert league.present.tolist()[:4] == [
        True,
        False,
        False,
        False,
    ]  # before the first snapshot: scripted only
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.add_(1.0)
    averages.update(policy, adam_steps=1)  # the uniform average now holds an update
    add_snapshots(league, [policy], averages=averages)
    assert (
        league.present[[SCRIPTED, EMA, UNIFORM, FIRST_SNAPSHOT]].all()
        and league.present_count() == 4
    )
    name, _ = next(iter(policy.named_parameters()))
    for member, which in ((EMA, "ema"), (UNIFORM, "uniform")):
        slot = int(league.member_slot[member])
        assert torch.equal(
            league.slot_weights[slot][f"policy.{name}"], getattr(averages, which)[name]
        )
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.add_(1.0)
    averages.update(policy, adam_steps=2)
    league.after_update(
        policy, averages, 1, 2, torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
    )
    slot = int(league.member_slot[EMA])
    assert torch.equal(league.slot_weights[slot][f"policy.{name}"], averages.ema[name])
    assert league.snapshots == 1  # no snapshot was due


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
    policy = model.Policy(model.ModelConfig())
    averages = WeightAverages(policy, LearnConfig())
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
