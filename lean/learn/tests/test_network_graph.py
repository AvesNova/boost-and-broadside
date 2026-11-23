"""The league's network moves replayed as one CUDA graph (rollout.NetworkGraph) against the
per-slot calls it captures. GPU only; slow (the routing graph compiles).

The networks' carries must match exactly (the same kernels). The sampled actions come from
the same seed but not always the same draws: on the laptop the first replay after a capture
differed from the per-slot calls in 3 of 96 entries and the next replay in none (CUDA
graphs advance the generator their own way); so the actions are compared as mostly equal."""

import dataclasses

import pytest
import torch

from lean import game, model, scripted
from lean.learn import League, LearnConfig, Opponent, WeightAverages, start
from lean.learn.compiled import graphs
from lean.learn.pool import EMA, FIRST_SNAPSHOT, UNIFORM
from lean.learn.rollout import NetworkGraph, act, league_actions, map_fields, network_actions
from lean.learn.tests.setting import add_snapshots

pytestmark = [pytest.mark.slow, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")]


def test_the_replayed_graph_equals_the_per_slot_calls_and_follows_the_table():
    device = torch.device("cuda")
    torch.manual_seed(0)
    config = LearnConfig(team_size_minimum=2, team_size_maximum=2)
    game_config = game.GameConfig(training_spawn=True)
    policy = model.Policy(model.ModelConfig()).to(device)
    league = League(config, device)
    opponent = Opponent(scripted.ScriptedConfig(), league)
    state, carry = start(config, game_config, policy, opponent, 16, torch.Generator(device).manual_seed(0))
    averages = WeightAverages(policy, config)
    averages.update(policy, 10**9)
    add_snapshots(league, [policy, model.Policy(model.ModelConfig()).to(device)], averages=averages)
    chosen = torch.tensor([0, EMA, UNIFORM, FIRST_SNAPSHOT, FIRST_SNAPSHOT + 1, 0, EMA, FIRST_SNAPSHOT], device=device)
    carry = dataclasses.replace(carry, member=chosen, slot=league.member_slot[chosen])
    league.prepare(8, carry.slot)
    step = graphs().slot_actions
    graph = NetworkGraph()

    def tick_inputs():
        # Whole tensors, as the compiled act gives (a fresh state's views can broadcast).
        views, *_, scripted_action, _, _, _, _ = act(config, game_config, policy, opponent, state, carry, None)
        return tuple(map_fields(view, torch.Tensor.contiguous) for view in views), scripted_action

    def moves(carry, with_graph: bool):
        views, scripted_action = tick_inputs()
        torch.cuda.manual_seed(1)
        if with_graph:
            return scripted_action, graph(config, league, views, carry, scripted_action, step)
        return scripted_action, network_actions(config, league, views, carry, scripted_action, None, step)

    moves(carry, True)  # the capture
    first_table = graph.table
    for _ in range(2):  # replays, the inputs copied in afresh each time
        scripted_action, (action, opponent_carry) = moves(carry, True)
        _, (eager_action, eager_carry) = moves(carry, False)
        assert (action == eager_action).float().mean() > 0.9
        assert torch.equal(opponent_carry, eager_carry)
        scripted_rows = chosen == 0
        assert torch.equal(action[scripted_rows], scripted_action[8:][scripted_rows])
    # A new table (the next segment's) is captured again; the replay follows it.
    league.prepare(8, carry.slot)
    moves(carry, True)  # captured again (the capture's own draws come after the seed)
    assert graph.table is league.table is not first_table
    scripted_action, (action, _) = moves(carry, True)
    _, (eager_action, _) = moves(carry, False)
    assert (action == eager_action).float().mean() > 0.9
    # And league_actions takes the graph when given one.
    views, scripted_action = tick_inputs()
    torch.cuda.manual_seed(1)
    through, _ = league_actions(config, opponent, views, carry, scripted_action, None, step, graph)
    assert (through == eager_action).float().mean() > 0.9
