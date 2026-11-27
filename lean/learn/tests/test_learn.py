import math

import pytest
import torch

from lean import game, model, scripted
from lean.contract import NO_PENDING, SHIP_FEATURES
from lean.learn import (
    LearnConfig,
    Opponent,
    collect,
    fallen_team_values,
    generalised_advantages,
    make_optimiser,
    normalise_advantages,
    start,
    update,
)
from lean.learn import rollout as rollout_module
from lean.learn.config import LEVELS, OUTCOME
from lean.learn.tests.setting import SMALL, add_snapshots, small


def test_gae_matches_hand_computed_case_and_stops_at_termination():
    # One environment, one ship, one level; discount 0.5, lambda 0.5; the episode ends on tick 1.
    reward = torch.tensor([1.0, 2.0, 3.0]).view(3, 1, 1, 1)
    value = torch.tensor([0.5, 1.0, 2.0]).view(3, 1, 1, 1)
    bootstrap = torch.tensor([4.0]).view(1, 1, 1)
    done = torch.tensor([False, True, False]).view(3, 1)
    advantage, target = generalised_advantages(
        reward, value, bootstrap, done, torch.tensor([0.5]), 0.5
    )
    # tick 2: 3 + 0.5 * 4 - 2 = 3; tick 1 (terminal): 2 - 1 = 1; tick 0: (1 + 0.5 * 1 - 0.5) + 0.25
    # * 1 = 1.25
    assert torch.allclose(advantage.flatten(), torch.tensor([1.25, 1.0, 3.0]))
    assert torch.allclose(target.flatten(), torch.tensor([1.75, 2.0, 5.0]))


def test_gae_scan_equals_the_tick_by_tick_recursion():
    """The reverse linear scan against the plain backward loop, per level, with episode ends."""
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


def perfect_critic_targets(before_end: int, fill: bool) -> tuple[float, float, float]:
    """Two team-mates and one enemy, a +10 outcome on the last of 512 ticks, a perfect
    outcome critic on the living rows, ship 1 dead for the last `before_end` ticks. Returns
    (survivor's target, the fallen ship's target, both at its last living tick; the ideal
    10 gamma^k)."""
    ticks, discount, lam = 512, 0.9997, 0.97
    death = ticks - before_end
    perfect = 10.0 * discount ** (ticks - 1 - torch.arange(ticks, dtype=torch.float64))
    own = torch.ones(ticks, 1, 3, dtype=torch.bool)
    own[:, 0, 2] = False  # the enemy row
    own[death:, 0, 1] = False
    fallen = torch.zeros_like(own)
    fallen[death:, 0, 1] = True
    value = perfect[:, None, None, None].expand(ticks, 1, 3, 1) * own[..., None]
    if fill:
        value = fallen_team_values(value, own, fallen)
    reward = torch.zeros(ticks, 1, 3, 1, dtype=torch.float64)
    reward[-1, 0, :2] = 10.0
    reward[-1, 0, 2] = -10.0
    done = torch.zeros(ticks, 1, dtype=torch.bool)
    done[-1] = True
    _, target = generalised_advantages(
        reward,
        value,
        torch.zeros(1, 3, 1, dtype=torch.float64),
        done,
        torch.tensor([discount], dtype=torch.float64),
        lam,
    )
    row = target[death - 1, 0, :, 0]
    return row[0].item(), row[1].item(), 10.0 * discount ** (ticks - death)


@pytest.mark.parametrize("before_end", [30, 100, 300])
def test_a_fallen_ship_keeps_the_outcome_at_the_discount(before_end):
    """The review's perfect-critic check (correctness-review-2026-11-28 finding 1): with its
    team-mates' value after death, a ship dying 30, 100 or 300 ticks before a +10 end keeps
    10 gamma^k at its last living tick (with zero values it fell to 3.97, 0.46, 0.001);
    the survivor's target is unchanged by the fill."""
    survivor, fallen_ship, ideal = perfect_critic_targets(before_end, fill=True)
    unfilled_survivor, unfilled_fallen, _ = perfect_critic_targets(before_end, fill=False)
    assert math.isclose(fallen_ship, ideal, rel_tol=1e-9)
    assert math.isclose(survivor, ideal, rel_tol=1e-9)
    assert survivor == unfilled_survivor
    assert unfilled_fallen < 0.5 * ideal


def test_fallen_team_values_fill_only_fallen_rows_with_the_living_mean():
    generator = torch.Generator().manual_seed(0)
    ticks, envs, slots = 5, 3, 4
    own = torch.rand(ticks, envs, slots, generator=generator) < 0.5
    fallen = ~own & (torch.rand(ticks, envs, slots, generator=generator) < 0.5)
    value = torch.randn(ticks, envs, slots, LEVELS, generator=generator) * own[..., None]
    filled = fallen_team_values(value, own, fallen)
    assert torch.equal(filled[~fallen], value[~fallen])
    count = own.sum(-1, keepdim=True).clamp(min=1)[..., None]
    mean = (value.sum(-2, keepdim=True) / count).expand_as(value)
    assert torch.allclose(filled[fallen], mean[fallen])


def test_normalised_advantages_keep_their_mean_and_have_unit_standard_deviation():
    generator = torch.Generator().manual_seed(0)
    advantage = torch.randn(16, 8, 4, 4, generator=generator) + 0.5
    mask = torch.rand(16, 8, 4, generator=generator) < 0.7
    normalised = normalise_advantages(advantage.sum(-1), mask)
    chosen = normalised[mask]
    assert (normalised[~mask] == 0).all()
    assert chosen.mean() > 0.2  # the mean (2.0 before scaling) is not removed
    assert torch.isclose(chosen.std(), torch.tensor(1.0), atol=1e-4)


ENVIRONMENTS = 8  # half self-play (the tests' layout, setting.SMALL): 12 seats
SEATS = ENVIRONMENTS + round(SMALL["self_play_share"] * ENVIRONMENTS)
CLOCK = 6  # every game ends at the clock on its sixth tick


def run_updates(updates, snapshot=False):
    config, game_config, policy, opponent, state, carry, generator = small(
        ENVIRONMENTS,
        game_config=game.GameConfig(training_spawn=True, clock_ticks=CLOCK),
        league=snapshot,
        team_size_minimum=1,
        equal_teams=False,
        epochs=2,
    )
    if snapshot:
        add_snapshots(opponent.league, [model.Policy(model.ModelConfig())])
    optimiser = make_optimiser(config, policy)
    history, steps = [], 0
    for _ in range(updates):
        rollout = collect(config, game_config, policy, opponent, state, carry, 16, generator)
        state, carry = rollout.state, rollout.carry
        history.append(
            update(
                config,
                policy,
                optimiser,
                rollout,
                teacher_weight=2.0,
                step_index=steps,
                generator=generator,
            )
        )
        steps += history[-1]["adam_steps"]
    return rollout, history


def test_rollout_shapes_and_outcome_reward_reach_both_teams():
    rollout, history = run_updates(1, snapshot=True)
    assert rollout.reward.shape == (16, SEATS, 4, LEVELS)
    assert rollout.observation.ship.shape == (16, SEATS, 4, SHIP_FEATURES)
    # Every game ends at ticks 5 and 11, for both seats of a self-play game.
    assert rollout.done[[CLOCK - 1, 2 * CLOCK - 1]].all() and rollout.done.sum() == 2 * SEATS
    assert (rollout.value[~rollout.observation.own] == 0).all()
    assert torch.isfinite(rollout.log_probability).all()
    outcome = rollout.reward[..., OUTCOME][rollout.done]  # (ended episodes, N)
    assert (outcome > 0).any() and (outcome < 0).any()  # winners and losers are both paid
    assert rollout.position_offset.shape == (16, SEATS, 2)
    assert not rollout.next_valid[rollout.done].any()
    assert not (rollout.fallen & rollout.observation.own).any()
    assert all(value == value for value in history[0].values())


def test_rl_updates_with_the_teacher_term_reduce_its_cross_entropy():
    _, history = run_updates(6)
    assert all(torch.isfinite(torch.tensor(list(metrics.values()))).all() for metrics in history)
    assert history[-1]["bc"] < history[0]["bc"]


def test_collect_pairs_actions_and_log_probabilities_under_the_action_delay():
    """Guards the one-tick action delay in the rollout: each seat's recorded action is the
    policy's sample on that tick's observation (its log-probability recomputed from the
    observations matches), and it is the pending action the next tick's observation shows
    (the action the game applies then), for own ships alive on both ticks within an episode."""
    config, game_config, policy, opponent, state, carry, generator = small(
        game_config=game.GameConfig(
            training_spawn=True, clock_ticks=10
        )  # an episode ends inside the rollout
    )
    rollout = collect(config, game_config, policy, opponent, state, carry, 16, generator)
    observation, own, action = rollout.observation, rollout.observation.own, rollout.action
    assert rollout.reset[1:].any()
    logits, *_ = policy.sequence(
        observation, rollout.initial_carry, rollout.reset, rollout.position_offset
    )
    recomputed = torch.log_softmax(logits.float(), -1).gather(-1, action[..., None])[..., 0]
    assert torch.allclose(recomputed[own], rollout.log_probability[own], atol=1e-4)
    carried = own[:-1] & own[1:] & ~rollout.reset[1:, :, None]
    assert carried.sum() > 50
    assert torch.equal(observation.pending[1:][carried], action[:-1][carried])
    assert (observation.pending[~own] == NO_PENDING).all()  # enemies, dead and absent rows


def test_next_state_loss_is_live():
    config, game_config, policy, opponent, state, carry, generator = small()
    rollout = collect(config, game_config, policy, opponent, state, carry, 8, generator)
    assert rollout.next_state.shape == (
        8,
        6,
        4,
        policy.config.next_state_size,
    )  # 4 environments, 2 self-play: 6 seats
    assert rollout.next_valid.any()
    assert torch.isfinite(rollout.next_state[rollout.next_valid]).all()
    assert rollout.next_state[rollout.next_valid].abs().sum() > 0
    metrics = update(config, policy, make_optimiser(config, policy), rollout, generator=generator)
    # nats per column: a fresh head (last layer orthogonal 0.01) is near uniform over its bins
    assert abs(metrics["next_state"] - math.log(policy.config.bins)) < 0.05


def test_a_next_state_width_mismatch_raises():
    config, game_config, policy, opponent, state, carry, generator = small()
    rollout = collect(config, game_config, policy, opponent, state, carry, 4, generator)
    rollout.next_state = rollout.next_state[..., :1]
    with pytest.raises(ValueError):
        update(config, policy, make_optimiser(config, policy), rollout, generator=generator)


def test_targets_once_per_update_equal_the_first_epochs_fresh_targets():
    """targets_every_epoch=False takes the values stored at collection; before any step they
    equal the critic's recomputed values, so both paths give the same first targets."""
    from lean.learn.store import RolloutStore
    from lean.learn.training import target_pass

    config, game_config, policy, opponent, state, carry, generator = small(
        epochs=2, micro_batch_views=4
    )
    rollout = collect(config, game_config, policy, opponent, state, carry, 8, generator)
    store = RolloutStore.of(rollout)
    index, _ = store.batch_index(torch.arange(store.views), 4)
    stored_flags, stored_sums = target_pass(config, policy, store, index, None)
    stored = store.advantage.clone(), store.target.clone()
    fresh_flags, fresh_sums = target_pass(config, policy, store, index)
    # The policy's step (stored) and sequence (recomputed) agree within float32 rounding of the
    # model.
    assert torch.allclose(stored[0], store.advantage, atol=1e-3) and torch.allclose(
        stored[1], store.target, atol=1e-3
    )
    assert torch.equal(stored_flags, fresh_flags)
    assert torch.allclose(stored_sums, fresh_sums, rtol=1e-3, atol=1e-3)
    for every_epoch in (False, True):
        variant = LearnConfig(
            team_size_minimum=2,
            team_size_maximum=2,
            minibatches=2,
            epochs=2,
            targets_every_epoch=every_epoch,
        )
        metrics = update(
            variant, policy, make_optimiser(variant, policy), rollout, generator=generator
        )
        assert all(value == value for value in metrics.values())


def test_finished_environments_take_the_segments_deals_in_order():
    """Agreed fixes 7 and 9: deals (game deal, team sizes, personas, learner team, position
    offset) are drawn once per segment; the k-th environment to finish takes deal k."""
    generator = torch.Generator().manual_seed(0)
    config = LearnConfig(
        team_size_minimum=2, team_size_maximum=4, equal_teams=False, persona_spread=1.0
    )
    game_config = game.GameConfig()
    policy = model.Policy(model.ModelConfig())
    opponent = Opponent(scripted.ScriptedConfig())
    envs = 4
    state, carry = start(config, game_config, policy, opponent, envs, generator)
    pool = rollout_module.draw_next_episodes(
        config, game_config, policy, opponent, envs, state.x.shape[1], generator, "cpu"
    )
    carry.next_episodes = pool
    taken_by = {}
    for done in ([False, True, False, True], [True, False, False, False]):
        done = torch.tensor(done)
        before = state
        state, carry = rollout_module.redeal(opponent, state, carry, done, generator)
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


def test_a_segment_uses_deals_again_only_past_one_per_environment():
    """More games ending in one segment than there are environments reuse deals (modulo E)."""
    generator = torch.Generator().manual_seed(0)
    config = LearnConfig(team_size_minimum=1, team_size_maximum=1)
    policy = model.Policy(model.ModelConfig())
    opponent = Opponent(scripted.ScriptedConfig())
    for clock in (
        10,
        4,
    ):  # 12 ticks: each of 4 games ends once, or three times (8 deals used twice)
        game_config = game.GameConfig(clock_ticks=clock)
        state, carry = start(config, game_config, policy, opponent, 4, generator)
        rollout = collect(config, game_config, policy, opponent, state, carry, 12, generator)
        assert int(rollout.done[:, :4].sum()) == 4 * (
            12 // clock
        )  # seats 0..3 are the environments
        assert int(rollout.carry.next_episodes.taken) == 4 * (12 // clock)


@pytest.mark.parametrize("chunks", [2, 5])
def test_the_chunked_policy_step_equals_one_call(chunks):
    """rollout.chunked_step (the live policy's seats in chunks, LearnConfig.policy_step_chunks):
    the rows are independent, so the joined outputs equal one call's to float32 rounding, also
    when the chunks do not divide the rows (12 rows into 5: 3, 3, 3, 3)."""
    torch.manual_seed(0)
    game_config = game.GameConfig(training_spawn=True)
    envs = 12
    state = game.reset(
        game_config, envs, torch.full((envs, 2), 4), torch.Generator().manual_seed(0), "cpu"
    )
    view = game.observe(game_config, state, 0)
    policy = model.Policy(model.ModelConfig.for_arena(game_config.arena_radius))
    carry = torch.randn(policy.initial_carry(envs, state.exists.shape[1], "cpu").shape)
    offset = torch.randn(envs, 2) * 100
    reset = torch.rand(envs) < 0.3
    whole = policy.step(view, carry, offset, reset)
    parts = rollout_module.chunked_step(policy, chunks)(view, carry, offset, reset)
    for joined, one in zip(parts, whole, strict=True):
        assert joined.shape == one.shape
        torch.testing.assert_close(joined, one, rtol=1e-5, atol=1e-5 * one.abs().max().item())
