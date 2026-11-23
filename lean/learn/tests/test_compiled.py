"""The compiled training path (lean.learn.compiled) on the CPU: it traces, and the compiled
loss equals the eager one. Slow: every graph compiles."""

import copy

import pytest
import torch

from lean import game, model, scripted
from lean.learn import League, LearnConfig, Opponent, collect, make_optimiser, start, update

pytestmark = pytest.mark.slow


def test_compiled_collect_and_update_run_with_the_real_packages():
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    config = LearnConfig(team_size_minimum=2, team_size_maximum=2, minibatches=2, epochs=1)
    game_config = game.GameConfig(ghost_bullets=True)
    policy = model.Policy(model.ModelConfig())
    league = League(LearnConfig(snapshot_every=1, snapshot_capacity=2))
    league.maybe_add(policy, 1)
    opponent = Opponent(scripted.ScriptedConfig(), league)
    state, carry = start(config, game_config, policy, opponent, 8, generator)
    for _ in range(2):  # the second rollout reuses the graphs
        rollout = collect(config, game_config, policy, opponent, state, carry, 6, generator, compiled=True)
        assert rollout.action.shape[:2] == (6, 12)  # 8 environments, 4 self-play: 12 seats
        state, carry = rollout.state, rollout.carry
        metrics = update(config, policy, make_optimiser(config, policy), rollout, generator=generator, compiled=True)
        assert all(value == value for value in metrics.values())


def test_compiled_loss_matches_eager():
    from lean.learn import RolloutStore
    from lean.learn.compiled import graphs
    from lean.learn.training import LOSS_NAMES, minibatch_loss, target_pass

    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    config = LearnConfig(team_size_minimum=2, team_size_maximum=2)
    game_config = game.GameConfig(ghost_bullets=True)
    policy = model.Policy(model.ModelConfig())
    opponent = Opponent(scripted.ScriptedConfig())
    state, carry = start(config, game_config, policy, opponent, 4, generator)
    rollout = collect(config, game_config, policy, opponent, state, carry, 6, generator)
    store = RolloutStore.of(rollout)
    index, _ = store.batch_index(torch.arange(store.views), store.views)
    target_pass(config, policy, store, index, None)
    (batch,) = list(store.batches(store.observation_names + LOSS_NAMES, index))
    views = batch["views"]
    observation, offset = store.observation(batch)
    arguments = (
        observation, batch["carry"], batch["reset"], offset, batch["action"], batch["log_probability"],
        store.advantage.index_select(1, views), store.target.index_select(1, views), batch["teacher"],
        batch["next_state"], batch["next_valid"], batch["hit_label"], torch.tensor(1.0),
        store.counts().index_select(0, views).sum(0),
    )
    twin = copy.deepcopy(policy)
    eager_loss, eager_measures = minibatch_loss(config, policy, *arguments)
    compiled_loss, compiled_measures = graphs().minibatch_loss(config, twin, *arguments)
    assert torch.allclose(eager_loss, compiled_loss, rtol=1e-4, atol=1e-5)
    for name in eager_measures:
        assert torch.allclose(eager_measures[name], compiled_measures[name], rtol=1e-4, atol=1e-5), name
    eager_loss.backward()
    compiled_loss.backward()
    for (name, eager), compiled_parameter in zip(policy.named_parameters(), twin.parameters(), strict=True):
        if eager.grad is None:
            assert compiled_parameter.grad is None or not compiled_parameter.grad.any(), name
            continue
        assert torch.allclose(eager.grad, compiled_parameter.grad, rtol=1e-3, atol=1e-5), name
