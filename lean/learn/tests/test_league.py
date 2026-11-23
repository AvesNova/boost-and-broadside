"""The league design (author, 2026-11-22): half self-play with both sides training, half
league games with only the live side training; the scripted agent is a member; PFSP
(1 - p)^2 with a 1/8 uniform floor; AlphaStar's decaying tallies; redraws only at reset."""

import dataclasses

import torch

from lean import game, model
from lean.contract import split_joint
from lean.learn import League, LearnConfig, abandon, collect, make_optimiser, update
from lean.learn.rollout import SEAT_AXIS, seat_view, snapshot_actions
from lean.learn.tests.setting import StepRecorder, small

ENVS = 8  # 4 self-play, 4 league


def league_of(capacity: int, snapshots: int = 0) -> League:
    league = League(LearnConfig(snapshot_every=1, snapshot_capacity=capacity))
    for _ in range(snapshots):
        league.maybe_add(model.Policy(model.ModelConfig()), 0)
    return league


def test_self_play_trains_both_teams_and_league_games_only_the_live_side():
    config, game_config, policy, opponent, state, carry, generator = small(ENVS, league_capacity=2)
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
    assert any(not torch.equal(old, new) for old, new in zip(before, policy.parameters()))


def test_both_sides_of_a_self_play_game_are_the_live_policy(monkeypatch):
    """The game applies the live policy's samples to both teams of a self-play game, and to
    the live team of a league game, whose other team plays the scripted agent's action
    (the scripted member, the only one before a snapshot)."""
    recorder = StepRecorder().install(monkeypatch)
    config, game_config, policy, opponent, state, carry, generator = small(ENVS, league_capacity=2)
    rollout = collect(config, game_config, policy, opponent, state, carry, 6, generator)
    team = state.team
    for tick, applied in enumerate(recorder.actions):
        for environment in range(ENVS):
            live = team[environment] == rollout.carry.learner_team[environment]  # no game ended
            assert torch.equal(applied[environment][live], split_joint(rollout.action[tick, environment])[live])
            if environment < ENVS // 2:
                other = split_joint(rollout.action[tick, ENVS + environment])
            else:
                other = split_joint(rollout.teacher[tick, environment])
            assert torch.equal(applied[environment][~live], other[~live])


def test_the_scripted_agent_is_drawn_as_a_member_next_to_snapshots():
    league = league_of(2, snapshots=2)
    config, game_config, policy, opponent, state, carry, generator = small(
        16, game_config=game.GameConfig(training_spawn=True, clock_ticks=4)
    )
    opponent.league = league
    carry.member = league.draw(8, generator)
    rollout = collect(config, game_config, policy, opponent, state, carry, 24, generator)
    assert int(rollout.done[:, 8:16].sum()) == 48  # league games
    assert (league.games > 0).all()  # every member was drawn and played
    # Drawn uniformly at first, the scripted agent is a third of the draws.
    draws = league_of(2, snapshots=2).draw(30000, torch.Generator().manual_seed(1))
    assert abs((draws == 0).float().mean().item() - 1 / 3) < 0.015


def test_sampling_weights_are_squared_loss_rate_with_an_eighth_uniform():
    league = league_of(3, snapshots=3)
    # Fixed tallies: win rates 0.2, 0.5 (draws count half), 0.9, and member 3 unplayed (0.5).
    league.games = torch.tensor([10.0, 4.0, 10.0, 0.0])
    league.wins = torch.tensor([2.0, 1.0, 9.0, 0.0])
    league.draws = torch.tensor([0.0, 2.0, 0.0, 0.0])
    assert torch.allclose(league.win_rates(), torch.tensor([0.2, 0.5, 0.9, 0.5]))
    weight = torch.tensor([0.64, 0.25, 0.01, 0.25])
    expected = 7 / 8 * weight / weight.sum() + 1 / 8 * 0.25
    assert torch.allclose(league.probabilities(), expected)
    samples = 200000
    frequency = torch.bincount(league.draw(samples, torch.Generator().manual_seed(2)), minlength=4).float() / samples
    sigma = (expected * (1 - expected) / samples).sqrt()
    assert ((frequency - expected).abs() < 5 * sigma).all()
    # The floor keeps a member beaten every time in play; absent slots are never drawn.
    league.wins[2] = 10.0
    assert league.probabilities()[2] >= 1 / 8 / 4 - 1e-6
    assert torch.allclose(league_of(3, snapshots=1).probabilities(), torch.tensor([0.5, 0.5, 0.0, 0.0]))


def test_tallies_decay_by_0_99_per_game_and_unplayed_reads_one_half():
    league = league_of(1, snapshots=1)
    assert torch.allclose(league.win_rates(), torch.tensor([0.5, 0.5]))
    member = torch.tensor([1])
    league.record(member, torch.tensor([True]), torch.tensor([True]), torch.tensor([False]))
    assert torch.isclose(league.games[1], torch.tensor(1.0)) and torch.isclose(league.win_rates()[1], torch.tensor(1.0))
    league.record(member, torch.tensor([True]), torch.tensor([False]), torch.tensor([False]))
    assert torch.isclose(league.games[1], torch.tensor(1.99))
    assert torch.isclose(league.wins[1], torch.tensor(0.99))
    assert torch.isclose(league.win_rates()[0], torch.tensor(0.5))  # the scripted agent, unplayed
    # Two games ending on one tick equal the two orders averaged.
    together = league_of(1, snapshots=1)
    together.record(torch.tensor([1, 1, 0]), torch.tensor([True, True, False]), torch.tensor([True, False, True]), torch.tensor([False, False, False]))
    assert torch.isclose(together.games[1], torch.tensor(1.99))
    assert torch.isclose(together.wins[1], torch.tensor((1.0 + 0.99) / 2))
    assert together.games[0] == 0  # unfinished games count nothing
    # A replaced member starts unplayed.
    league.maybe_add(model.Policy(model.ModelConfig()), 0)
    assert league.games[1] == 0 and torch.isclose(league.win_rates()[1], torch.tensor(0.5))


def test_opponents_are_redrawn_only_at_reset():
    config, game_config, policy, opponent, state, carry, generator = small(
        16, game_config=game.GameConfig(training_spawn=True, clock_ticks=5)
    )
    opponent.league = league_of(3, snapshots=3)
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


def test_abandon_redeals_only_the_games_of_a_replaced_member():
    config, game_config, policy, opponent, state, carry, generator = small(32)
    opponent.league = league = league_of(1, snapshots=1)
    carry.member = league.draw(16, generator)
    rollout = collect(config, game_config, policy, opponent, state, carry, 2, generator)
    state, carry = rollout.state, rollout.carry
    playing = carry.member == 1
    assert playing.any() and (~playing).any()
    replaced = league.maybe_add(policy, 0)
    state, after = abandon(config, game_config, policy, opponent, state, carry, replaced, generator)
    league_reset = after.reset[16:]
    assert (league_reset[playing]).all()
    assert not league_reset[~playing & ~carry.reset[16:]].any()
    assert not after.reset[:16][~carry.reset[:16]].any()
    assert (state.tick[16:][playing] == 0).all()


def test_snapshot_members_run_in_collect_and_update():
    config, game_config, policy, opponent, state, carry, generator = small(6, league_capacity=2)
    opponent.league.maybe_add(policy, 0)
    carry.member = torch.tensor([0, 1, 1])  # environments 3..5 are league games
    before = carry.opponent.clone()
    rollout = collect(config, game_config, policy, opponent, state, carry, 4, generator)
    after = rollout.carry.opponent
    assert torch.equal(after[:, 0], before[:, 0])  # the scripted member's environment
    assert not torch.equal(after[:, 1:], before[:, 1:])  # the snapshot ran there
    metrics = update(config, policy, make_optimiser(config, policy), rollout, generator=generator)
    assert all(value == value for value in metrics.values())


def test_batched_snapshots_give_each_environment_its_own_members_step():
    """snapshot_actions runs every slot on every league environment at once; each
    environment must get the carry its own member's step gives, and scripted environments
    keep theirs and take the scripted action."""
    torch.manual_seed(0)
    members = [model.Policy(model.ModelConfig()) for _ in range(3)]
    league = league_of(3)
    for member in members:
        league.maybe_add(member, 0)
    config, game_config, policy, opponent, state, carry, generator = small(12)
    opponent.league = league
    carry.member = torch.tensor([0, 1, 2, 3, 1, 3])  # the 6 league environments
    carry = dataclasses.replace(carry, opponent=torch.randn_like(carry.opponent))
    views = (game.observe(game_config, state, 0), game.observe(game_config, state, 1))
    scripted_action = torch.full((12, state.exists.shape[1], 3), 1)
    parameters, buffers = league.weights
    action, new_carry = snapshot_actions(config, league.template, parameters, buffers, views, carry, scripted_action, generator)
    first = 6
    view = seat_view(views, torch.arange(first, 12), 1 - carry.learner_team[first:])
    for row, member in enumerate(carry.member.tolist()):
        if member == 0:
            assert (action[row] == 1).all()
            assert torch.equal(new_carry.select(SEAT_AXIS, row), carry.opponent.select(SEAT_AXIS, row))
            continue
        _, _, expected = members[member - 1].step(view, carry.opponent, carry.position_offset[first:])
        # vmap runs the math attention kernel, the plain step the fused one: float32 rounding apart.
        assert torch.allclose(new_carry.select(SEAT_AXIS, row), expected.select(SEAT_AXIS, row), atol=1e-4)
