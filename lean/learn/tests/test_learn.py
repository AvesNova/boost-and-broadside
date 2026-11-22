import pytest
import torch

from lean.learn import (
    League,
    LearnConfig,
    Opponent,
    bc_weight,
    collect,
    generalised_advantages,
    make_optimiser,
    normalise_advantages,
    start,
    update,
)
from lean.learn.tests import standins


def test_gae_matches_hand_computed_case_and_stops_at_termination():
    # One environment, one ship, one level; discount 0.5, lambda 0.5; the episode ends on tick 1.
    reward = torch.tensor([1.0, 2.0, 3.0]).view(3, 1, 1, 1)
    value = torch.tensor([0.5, 1.0, 2.0]).view(3, 1, 1, 1)
    bootstrap = torch.tensor([4.0]).view(1, 1, 1)
    done = torch.tensor([False, True, False]).view(3, 1)
    advantage, target = generalised_advantages(reward, value, bootstrap, done, torch.tensor([0.5]), 0.5)
    # tick 2: 3 + 0.5 * 4 - 2 = 3; tick 1 (terminal): 2 - 1 = 1; tick 0: (1 + 0.5 * 1 - 0.5) + 0.25 * 1 = 1.25
    assert torch.allclose(advantage.flatten(), torch.tensor([1.25, 1.0, 3.0]))
    assert torch.allclose(target.flatten(), torch.tensor([1.75, 2.0, 5.0]))


@pytest.mark.parametrize("normaliser", ["std", "percentile"])
def test_normalised_advantages_keep_their_mean_and_have_unit_scale(normaliser):
    generator = torch.Generator().manual_seed(0)
    advantage = torch.randn(16, 8, 4, 4, generator=generator) + 0.5
    mask = torch.rand(16, 8, 4, generator=generator) < 0.7
    config = LearnConfig(advantage_normaliser=normaliser)
    normalised = normalise_advantages(advantage, mask, config)
    chosen = normalised[mask]
    assert (normalised[~mask] == 0).all()
    assert chosen.mean() > 0.2  # the mean (2.0 before scaling) is not removed
    if normaliser == "std":
        assert torch.isclose(chosen.std(), torch.tensor(1.0), atol=1e-4)
    else:
        assert torch.isclose(chosen.quantile(0.95) - chosen.quantile(0.05), torch.tensor(1.0), atol=1e-4)


def test_bc_gate_is_full_at_zero_win_rate_and_off_at_the_gate():
    config = LearnConfig(bc_coefficient=2.0, bc_gate=0.45)
    assert bc_weight(config, 0.0) == 2.0
    assert bc_weight(config, 0.45) == 0.0
    assert bc_weight(config, 0.9) == 0.0
    assert 0.0 < bc_weight(config, 0.2) < 2.0


def run_updates(updates, snapshot=False):
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    config = LearnConfig(team_size_minimum=1, team_size_maximum=2, equal_teams=False, minibatches=2, epochs=2, warmup_updates=1)
    game_config = standins.GameConfig()
    policy = standins.TinyPolicy()
    league = None
    if snapshot:
        league = League(every=1, capacity=2)
        league.maybe_add(standins.TinyPolicy(), 0)
    opponent = Opponent(scripted_config=None, league=league)
    optimiser = make_optimiser(config, policy)
    state, carry = start(config, game_config, policy, opponent, 8, generator, game=standins.game, scripted=standins.scripted)
    history = []
    for index in range(updates):
        rollout = collect(config, game_config, policy, opponent, state, carry, 16, generator, game=standins.game, scripted=standins.scripted)
        state, carry = rollout.state, rollout.carry
        history.append(update(config, policy, optimiser, rollout, update_index=index, generator=generator))
    return rollout, history


def test_rollout_shapes_and_outcome_reward_reach_both_teams():
    rollout, _ = run_updates(1, snapshot=True)
    # 8 environments, 4 of them self-play with two live seats: 12 seats.
    assert rollout.reward.shape == (16, 12, 4, 4)
    assert rollout.observation.ship.shape == (16, 12, 4, 3)
    assert int(rollout.episodes) == 16  # 8 environments, episodes of 6 ticks, 16 ticks
    assert rollout.done.sum() == 24  # each self-play episode ends for both of its seats
    assert (rollout.value[~rollout.observation.own] == 0).all()
    assert torch.isfinite(rollout.log_probability).all()
    outcome = rollout.reward[..., 0][rollout.done]  # (ended episodes, N)
    assert (outcome > 0).any() and (outcome < 0).any()  # winners and losers are both paid
    assert rollout.position_offset.shape == (16, 12, 2)
    assert not rollout.next_valid[rollout.done].any()


def test_updates_on_the_stand_in_reduce_the_bc_loss():
    _, history = run_updates(6)
    assert all(torch.isfinite(torch.tensor(list(metrics.values()))).all() for metrics in history)
    assert history[-1]["bc"] < history[0]["bc"]


def test_real_packages_smoke():
    game = pytest.importorskip("lean.game")
    scripted = pytest.importorskip("lean.scripted")
    model = pytest.importorskip("lean.model")
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    config = LearnConfig(team_size_minimum=2, team_size_maximum=2, minibatches=2, epochs=1)
    game_config = game.GameConfig()
    policy = model.Policy(model.ModelConfig())
    opponent = Opponent(scripted_config=scripted.ScriptedConfig())
    state, carry = start(config, game_config, policy, opponent, 4, generator)
    rollout = collect(config, game_config, policy, opponent, state, carry, 8, generator)
    metrics = update(config, policy, make_optimiser(config, policy), rollout, generator=generator)
    assert all(value == value for value in metrics.values())


def test_next_state_loss_is_live_with_the_real_packages():
    game = pytest.importorskip("lean.game")
    scripted = pytest.importorskip("lean.scripted")
    model = pytest.importorskip("lean.model")
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    config = LearnConfig(team_size_minimum=2, team_size_maximum=2, minibatches=2, epochs=1)
    policy = model.Policy(model.ModelConfig())
    opponent = Opponent(scripted_config=scripted.ScriptedConfig())
    state, carry = start(config, game.GameConfig(), policy, opponent, 4, generator)
    rollout = collect(config, game.GameConfig(), policy, opponent, state, carry, 8, generator)
    assert rollout.next_state.shape == (8, 6, 4, policy.config.next_state_size)  # 4 environments, 2 self-play: 6 seats
    assert rollout.next_valid.any()
    assert torch.isfinite(rollout.next_state[rollout.next_valid]).all()
    assert rollout.next_state[rollout.next_valid].abs().sum() > 0
    metrics = update(config, policy, make_optimiser(config, policy), rollout, generator=generator)
    assert metrics["next_state"] > 0 and metrics["next_state"] == metrics["next_state"] and metrics["next_state"] < float("inf")


def test_a_next_state_width_mismatch_raises():
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    config = LearnConfig(team_size_minimum=1, team_size_maximum=2, minibatches=2, epochs=1)
    policy = standins.TinyPolicy()
    opponent = Opponent(scripted_config=None)
    state, carry = start(config, standins.GameConfig(), policy, opponent, 4, generator, game=standins.game, scripted=standins.scripted)
    rollout = collect(config, standins.GameConfig(), policy, opponent, state, carry, 8, generator, game=standins.game, scripted=standins.scripted)
    rollout.next_state = rollout.next_state[..., :1]
    with pytest.raises(ValueError):
        update(config, policy, make_optimiser(config, policy), rollout, generator=generator)
