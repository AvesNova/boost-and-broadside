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
from lean.learn.config import LEVELS, OUTCOME
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


def test_gae_scan_equals_the_tick_by_tick_recursion():
    """The reverse doubling scan against the plain backward loop, per level, with episode ends."""
    generator = torch.Generator().manual_seed(0)
    ticks, envs, slots = 128, 6, 3
    reward = torch.randn(ticks, envs, slots, LEVELS, generator=generator)
    value = torch.randn(ticks, envs, slots, LEVELS, generator=generator)
    bootstrap = torch.randn(envs, slots, LEVELS, generator=generator)
    done = torch.rand(ticks, envs, generator=generator) < 0.02
    config = LearnConfig()
    discounts, lambdas = torch.tensor(config.discounts), torch.tensor(config.gae_lambdas)
    expected = torch.zeros_like(reward)
    running, next_value = torch.zeros_like(bootstrap), bootstrap
    for tick in range(ticks - 1, -1, -1):
        alive = (~done[tick]).float()[:, None, None]
        delta = reward[tick] + discounts * next_value * alive - value[tick]
        running = delta + discounts * lambdas * alive * running
        expected[tick], next_value = running, value[tick]
    advantage, target = generalised_advantages(reward, value, bootstrap, done, discounts, lambdas)
    assert torch.allclose(advantage, expected, rtol=1e-5, atol=1e-5)
    assert torch.allclose(target, expected + value, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("normaliser", ["std", "percentile"])
def test_normalised_advantages_keep_their_mean_and_have_unit_scale(normaliser):
    generator = torch.Generator().manual_seed(0)
    advantage = torch.randn(16, 8, 4, 4, generator=generator) + 0.5
    mask = torch.rand(16, 8, 4, generator=generator) < 0.7
    config = LearnConfig(advantage_normaliser=normaliser)
    normalised = normalise_advantages(advantage.sum(-1), mask, config)
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
    assert rollout.reward.shape == (16, 12, 4, LEVELS)
    assert rollout.observation.ship.shape == (16, 12, 4, 3)
    assert int(rollout.episodes) == 16  # 8 environments, episodes of 6 ticks, 16 ticks
    assert rollout.done.sum() == 24  # each self-play episode ends for both of its seats
    assert (rollout.value[~rollout.observation.own] == 0).all()
    assert torch.isfinite(rollout.log_probability).all()
    outcome = rollout.reward[..., OUTCOME][rollout.done]  # (ended episodes, N)
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


def test_targets_once_per_update_equal_the_first_epochs_fresh_targets():
    """targets_every_epoch=False takes the values stored at collection; before any step they
    equal the critic's recomputed values, so both paths give the same first targets."""
    from lean.learn.store import RolloutStore
    from lean.learn.update import target_pass

    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    config = LearnConfig(team_size_minimum=2, team_size_maximum=2, minibatches=2, epochs=2, micro_batch_views=4)
    policy = standins.TinyPolicy()
    opponent = Opponent(scripted_config=None)
    state, carry = start(config, standins.GameConfig(), policy, opponent, 4, generator, game=standins.game, scripted=standins.scripted)
    rollout = collect(config, standins.GameConfig(), policy, opponent, state, carry, 8, generator, game=standins.game, scripted=standins.scripted)
    store = RolloutStore.of(rollout)
    index, _ = store.batch_index(torch.arange(store.views), 4)
    stored_flags = target_pass(config, policy, store, index, None)
    stored = store.advantage.clone(), store.target.clone()
    fresh_flags = target_pass(config, policy, store, index)
    assert torch.allclose(stored[0], store.advantage, atol=1e-5) and torch.allclose(stored[1], store.target, atol=1e-5)
    assert torch.equal(stored_flags, fresh_flags)
    for every_epoch in (False, True):
        variant = LearnConfig(team_size_minimum=2, team_size_maximum=2, minibatches=2, epochs=2, targets_every_epoch=every_epoch)
        metrics = update(variant, policy, make_optimiser(variant, policy), rollout, generator=generator)
        assert all(value == value for value in metrics.values())


def test_finished_environments_take_the_segments_deals_in_order():
    """Agreed fixes 7 and 9: deals (game deal, team sizes, personas, learner team, position
    offset) are drawn once per segment; the k-th environment to finish takes deal k."""
    import importlib

    from lean import game, model, scripted

    collect_module = importlib.import_module("lean.learn.collect")
    generator = torch.Generator().manual_seed(0)
    config = LearnConfig(team_size_minimum=2, team_size_maximum=4, equal_teams=False)
    game_config = game.GameConfig.core()
    policy = model.Policy(model.ModelConfig())
    opponent = Opponent(scripted.ScriptedConfig(), persona_spread=1.0)
    envs = 4
    state, carry = start(config, game_config, policy, opponent, envs, generator)
    pool = collect_module.draw_next_episodes(
        config, game_config, policy, opponent, envs, state.x.shape[1], generator, "cpu", game, scripted
    )
    carry.next_episodes = pool
    taken_by = {}
    for done in ([False, True, False, True], [True, False, False, False]):
        done = torch.tensor(done)
        before = state
        state, carry = collect_module.redeal(opponent, state, carry, done, generator, game)
        for environment in done.nonzero().flatten().tolist():
            taken_by[environment] = len(taken_by)
            deal = taken_by[environment]
            assert torch.equal(state.ships[environment], pool.state.ships[deal])
            assert torch.equal(state.fields[environment], pool.state.fields[deal])
            assert torch.equal(state.seen[environment], pool.state.seen[deal])
            assert torch.equal(state.exists[environment], pool.state.exists[deal])
            assert int(carry.learner_team[environment]) == int(pool.learner_team[deal])
            assert torch.equal(carry.position_offset[environment], pool.position_offset[deal])
            for name in pool.persona:
                assert torch.equal(carry.persona[name][environment], pool.persona[name][deal]), name
        for environment in (~done).nonzero().flatten().tolist():
            assert torch.equal(state.ships[environment], before.ships[environment])
    assert taken_by == {1: 0, 3: 1, 0: 2}
    assert int(carry.next_episodes.taken) == 3


def test_a_segment_counts_deals_used_twice():
    """More games ending in one segment than there are environments reuse deals, counted."""
    from lean import game, model, scripted

    generator = torch.Generator().manual_seed(0)
    config = LearnConfig(team_size_minimum=1, team_size_maximum=1)
    policy = model.Policy(model.ModelConfig())
    opponent = Opponent(scripted.ScriptedConfig())
    for clock, repeated in ((10, 0), (4, 8)):  # 12 ticks: each of 4 games ends once, or three times
        game_config = game.GameConfig(clock_ticks=clock)
        state, carry = start(config, game_config, policy, opponent, 4, generator)
        rollout = collect(config, game_config, policy, opponent, state, carry, 12, generator)
        assert int(rollout.episodes) == 4 * (12 // clock)
        assert int(rollout.repeated_deals) == repeated
