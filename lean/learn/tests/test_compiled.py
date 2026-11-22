"""The compiled training path (lean.learn.compiled) on the CPU: it traces, and the compiled
loss equals the eager one. Slow: every graph compiles."""

import copy

import pytest
import torch

from lean.learn import League, LearnConfig, Opponent, collect, make_optimiser, start, update
from lean.learn.collect import index_carry, map_fields

pytestmark = pytest.mark.slow


def test_compiled_collect_and_update_run_with_the_real_packages():
    game = pytest.importorskip("lean.game")
    scripted = pytest.importorskip("lean.scripted")
    model = pytest.importorskip("lean.model")
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    config = LearnConfig(team_size_minimum=2, team_size_maximum=2, minibatches=2, epochs=1)
    game_config = game.GameConfig(ghost_bullets=True)
    policy = model.Policy(model.ModelConfig())
    league = League(1, 2)
    league.maybe_add(policy, 1)
    opponent = Opponent(scripted.ScriptedConfig(), league)
    state, carry = start(config, game_config, policy, opponent, 8, generator)
    for _ in range(2):  # the second rollout reuses the graphs
        rollout = collect(config, game_config, policy, opponent, state, carry, 6, generator, compiled=True)
        state, carry = rollout.state, rollout.carry
        metrics = update(config, policy, make_optimiser(config, policy), rollout, generator=generator, compiled=True)
        assert all(value == value for value in metrics.values())
    assert rollout.action.shape[:2] == (6, 12)  # 8 environments, 4 self-play: 12 seats


def test_compiled_loss_matches_eager():
    game = pytest.importorskip("lean.game")
    scripted = pytest.importorskip("lean.scripted")
    model = pytest.importorskip("lean.model")
    from lean.learn import compiled
    from lean.learn.update import fresh_targets, minibatch_loss

    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    config = LearnConfig(team_size_minimum=2, team_size_maximum=2)
    game_config = game.GameConfig(ghost_bullets=True)
    policy = model.Policy(model.ModelConfig())
    opponent = Opponent(scripted.ScriptedConfig())
    state, carry = start(config, game_config, policy, opponent, 4, generator)
    rollout = collect(config, game_config, policy, opponent, state, carry, 6, generator)
    index = torch.arange(rollout.done.shape[1])
    advantage, target, _ = fresh_targets(config, policy, rollout, [index])
    arguments = (
        map_fields(rollout.observation, lambda tensor: tensor[:, index]), index_carry(rollout.initial_carry, index),
        rollout.reset, rollout.position_offset, rollout.action, rollout.log_probability, advantage, target,
        rollout.teacher, rollout.next_state, rollout.next_valid, rollout.hit_label, torch.tensor(1.0),
    )
    twin = copy.deepcopy(policy)
    eager_loss, eager_measures = minibatch_loss(config, policy, *arguments)
    compiled_loss, compiled_measures = compiled.minibatch_loss(config, twin, *arguments)
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
