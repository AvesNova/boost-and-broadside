"""The league design (author, 2026-11-22): half self-play with both sides training, half
league games with only the live side training; the scripted agent is a member; PFSP
(1 - p)^2 with a 1/8 uniform floor; AlphaStar's decaying tallies; redraws only at reset."""

import pytest
import torch

from lean.learn import (
    League,
    LearnConfig,
    Opponent,
    abandon,
    collect,
    make_optimiser,
    start,
    update,
)
from lean.learn.tests import standins

ENVS = 8  # 4 self-play, 4 league


class NeverFires(standins.TinyPolicy):
    """A stand-in policy that never fires (odd joint actions fire): in the stand-in game a
    team scores only by firing, so two of these always draw and lose to the scripted
    agent, which always fires."""

    def __init__(self):
        super().__init__()
        with torch.no_grad():
            self.action.weight.zero_()
            self.action.bias.copy_(torch.tensor([0.0, -1e9] * 21))


def setup(policy, league, envs=ENVS, seed=0):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    config = LearnConfig(team_size_minimum=1, team_size_maximum=2, equal_teams=False, minibatches=2, epochs=1)
    opponent = Opponent(scripted_config=None, league=league)
    state, carry = start(config, standins.GameConfig(), policy, opponent, envs, generator, game=standins.game, scripted=standins.scripted)
    return config, opponent, state, carry, generator


def run(config, policy, opponent, state, carry, ticks, generator):
    return collect(config, standins.GameConfig(), policy, opponent, state, carry, ticks, generator, game=standins.game, scripted=standins.scripted)


def test_self_play_trains_both_teams_and_league_games_only_the_live_side():
    policy = standins.TinyPolicy()
    config, opponent, state, carry, generator = setup(policy, League(every=1, capacity=2))
    rollout = run(config, policy, opponent, state, carry, 6, generator)
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


def test_both_sides_of_a_self_play_game_are_the_live_policy():
    # A never-firing live policy: self-play games are all draws (both sides are the live
    # policy), league games against the scripted agent are all lost.
    policy = NeverFires()
    league = League(every=1, capacity=2)
    config, opponent, state, carry, generator = setup(policy, league)
    rollout = run(config, policy, opponent, state, carry, 6, generator)
    outcome = rollout.reward[-1, ..., 0] * rollout.observation.own[-1]  # every episode ends on tick 5
    assert rollout.done[-1].all()
    assert (outcome[rollout.self_play] == 0).all()  # draws
    assert (outcome[~rollout.self_play].sum(-1) < 0).all()  # losses
    assert int(rollout.league_games) == ENVS // 2 and int(rollout.league_wins) == 0
    assert int(rollout.scripted_games) == ENVS // 2  # only the scripted agent is a member yet
    assert torch.isclose(league.win_rates()[0], torch.tensor(0.0))


def test_the_scripted_agent_is_drawn_as_a_member_next_to_snapshots():
    policy = NeverFires()
    league = League(every=1, capacity=2)
    league.maybe_add(NeverFires(), 0)
    league.maybe_add(NeverFires(), 0)
    config, opponent, state, carry, generator = setup(policy, league, envs=64)
    rollout = run(config, policy, opponent, state, carry, 6 * 8, generator)
    # Against never-firing snapshots every game is a draw; against the scripted agent a loss.
    assert int(rollout.scripted_games) > 0
    assert int(rollout.league_games) > int(rollout.scripted_games)
    assert int(rollout.league_wins) == 0
    rates = league.win_rates()
    assert torch.isclose(rates[0], torch.tensor(0.0))
    assert torch.allclose(rates[1:], torch.tensor(0.5))  # all draws
    assert league.games[1:].min() > 0
    # Drawn uniformly at first, the scripted agent is a third of the draws.
    fresh = League(every=1, capacity=2)
    fresh.maybe_add(NeverFires(), 0)
    fresh.maybe_add(NeverFires(), 0)
    draws = fresh.draw(30000, torch.Generator().manual_seed(1))
    assert abs((draws == 0).float().mean().item() - 1 / 3) < 0.015


def test_sampling_weights_are_squared_loss_rate_with_an_eighth_uniform():
    league = League(every=1, capacity=3)
    for _ in range(3):
        league.maybe_add(standins.TinyPolicy(), 0)
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
    small = League(every=1, capacity=3)
    small.maybe_add(standins.TinyPolicy(), 0)
    assert torch.allclose(small.probabilities(), torch.tensor([0.5, 0.5, 0.0, 0.0]))


def test_tallies_decay_by_0_99_per_game_and_unplayed_reads_one_half():
    league = League(every=1, capacity=1)
    league.maybe_add(standins.TinyPolicy(), 0)
    assert torch.allclose(league.win_rates(), torch.tensor([0.5, 0.5]))
    member = torch.tensor([1])
    league.record(member, torch.tensor([True]), torch.tensor([True]), torch.tensor([False]))
    assert torch.isclose(league.games[1], torch.tensor(1.0)) and torch.isclose(league.win_rates()[1], torch.tensor(1.0))
    league.record(member, torch.tensor([True]), torch.tensor([False]), torch.tensor([False]))
    assert torch.isclose(league.games[1], torch.tensor(1.99))
    assert torch.isclose(league.wins[1], torch.tensor(0.99))
    assert torch.isclose(league.win_rates()[0], torch.tensor(0.5))  # the scripted agent, unplayed
    # Two games ending on one tick equal the two orders averaged.
    together = League(every=1, capacity=1)
    together.maybe_add(standins.TinyPolicy(), 0)
    together.record(torch.tensor([1, 1, 0]), torch.tensor([True, True, False]), torch.tensor([True, False, True]), torch.tensor([False, False, False]))
    assert torch.isclose(together.games[1], torch.tensor(1.99))
    assert torch.isclose(together.wins[1], torch.tensor((1.0 + 0.99) / 2))
    assert together.games[0] == 0  # unfinished games count nothing
    # A replaced member starts unplayed.
    league.maybe_add(standins.TinyPolicy(), 0)
    assert league.games[1] == 0 and torch.isclose(league.win_rates()[1], torch.tensor(0.5))


def test_opponents_are_redrawn_only_at_reset():
    policy = standins.TinyPolicy()
    league = League(every=1, capacity=3)
    for _ in range(3):
        league.maybe_add(standins.TinyPolicy(), 0)
    config, opponent, state, carry, generator = setup(policy, league, envs=32)
    changed_at_reset = 0
    for _ in range(24):
        before = carry.member.clone()
        rollout = run(config, policy, opponent, state, carry, 1, generator)
        state, carry = rollout.state, rollout.carry
        done = rollout.done[0, 16:32]  # seats 16..31 are the league environments
        changed = carry.member != before
        assert not (changed & ~done).any()
        changed_at_reset += int(changed.sum())
    assert changed_at_reset > 0


def test_abandon_redeals_only_the_games_of_a_replaced_member():
    policy = standins.TinyPolicy()
    league = League(every=1, capacity=1)
    league.maybe_add(standins.TinyPolicy(), 0)
    config, opponent, state, carry, generator = setup(policy, league, envs=32)
    rollout = run(config, policy, opponent, state, carry, 2, generator)
    state, carry = rollout.state, rollout.carry
    playing = carry.member == 1
    assert playing.any() and (~playing).any()
    replaced = league.maybe_add(policy, 0)
    state, after = abandon(config, standins.GameConfig(), policy, opponent, state, carry, replaced, generator, game=standins.game, scripted=standins.scripted)
    league_reset = after.reset[16:]
    assert (league_reset[playing]).all()
    assert not league_reset[~playing & ~carry.reset[16:]].any()
    assert not after.reset[:16][~carry.reset[:16]].any()
    assert (state.tick[16:][playing] == 0).all()


def test_snapshot_members_run_with_the_real_packages():
    game = pytest.importorskip("lean.game")
    scripted = pytest.importorskip("lean.scripted")
    model = pytest.importorskip("lean.model")
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    config = LearnConfig(team_size_minimum=2, team_size_maximum=2, minibatches=2, epochs=1)
    policy = model.Policy(model.ModelConfig())
    league = League(every=1, capacity=2)
    league.maybe_add(policy, 0)
    opponent = Opponent(scripted_config=scripted.ScriptedConfig(), league=league)
    state, carry = start(config, game.GameConfig(), policy, opponent, 6, generator)
    carry.member = torch.tensor([0, 1, 1])  # environments 3..5 are league games
    before = carry.opponent.clone()
    rollout = collect(config, game.GameConfig(), policy, opponent, state, carry, 4, generator)
    after = rollout.carry.opponent
    assert torch.equal(after[:, 0], before[:, 0])  # the scripted member's environment
    assert not torch.equal(after[:, 1:], before[:, 1:])  # the snapshot ran there
    metrics = update(config, policy, make_optimiser(config, policy), rollout, generator=generator)
    assert all(value == value for value in metrics.values())
