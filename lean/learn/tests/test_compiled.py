"""The compiled training path (lean.learn.compiled) on the CPU: it traces, and the compiled
loss equals the eager one. Slow: every graph compiles."""

import copy
import dataclasses

import pytest
import torch

from lean import game, model, scripted
from lean.learn import League, LearnConfig, Opponent, collect, make_optimiser, start, update
from lean.learn.compiled import graphs
from lean.learn.pool import EMA, FIRST_SNAPSHOT
from lean.learn.rollout import (
    act,
    advance,
    draw_next_episodes,
    league_actions,
    self_play_count,
    slot_actions,
)
from lean.learn.tests.setting import add_snapshots

pytestmark = pytest.mark.slow


def test_compiled_collect_and_update_run_with_the_real_packages():
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    config = LearnConfig(team_size_minimum=2, team_size_maximum=2, minibatches=2, epochs=1)
    game_config = game.GameConfig()
    policy = model.Policy(model.ModelConfig())
    league = League(config)
    opponent = Opponent(scripted.ScriptedConfig(), league)
    state, carry = start(config, game_config, policy, opponent, 8, generator)
    add_snapshots(league, [policy, model.Policy(model.ModelConfig())])
    # Every league game plays a network (the EMA or a snapshot), so the routing runs.
    chosen = torch.tensor([EMA, FIRST_SNAPSHOT, FIRST_SNAPSHOT + 1, FIRST_SNAPSHOT])
    carry = dataclasses.replace(carry, member=chosen, slot=league.member_slot[chosen])
    for _ in range(2):  # the second rollout reuses the graphs
        rollout = collect(
            config, game_config, policy, opponent, state, carry, 6, generator, compiled=True
        )
        seats = 8 + self_play_count(config, 8)  # 8 environments and their self-play seats
        assert rollout.action.shape[:2] == (6, seats)
        state, carry = rollout.state, rollout.carry
        metrics = update(
            config,
            policy,
            make_optimiser(config, policy),
            rollout,
            generator=generator,
            compiled=True,
        )
        assert all(value == value for value in metrics.values())


def test_a_static_graph_refuses_to_compile_again():
    """compile_shapes "static" (training): the first call compiles, a call at a new size
    raises instead of recompiling."""
    from lean.contract import compile_graph

    graph = compile_graph(lambda x: x * 2.0 + 1.0, "static")
    assert torch.equal(graph(torch.ones(3)), torch.full((3,), 3.0))
    assert torch.equal(graph(torch.zeros(3)), torch.ones(3))  # same size: no compile
    with pytest.raises(RuntimeError, match="recompile"):
        graph(torch.ones(4))


def test_compiled_loss_matches_eager():
    from lean.learn import RolloutStore
    from lean.learn.compiled import graphs
    from lean.learn.training import LOSS_NAMES, minibatch_loss, target_pass

    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    config = LearnConfig(team_size_minimum=2, team_size_maximum=2)
    game_config = game.GameConfig()
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
        observation,
        batch["carry"],
        batch["reset"],
        offset,
        batch["action"],
        batch["log_probability"],
        store.advantage.index_select(1, views),
        store.target.index_select(1, views),
        batch["teacher"],
        batch["next_state"],
        batch["next_valid"],
        batch["hit_label"],
        torch.tensor([1.0, 0.003, 0.5]),
        store.counts().index_select(0, views).sum(0),
    )
    twin = copy.deepcopy(policy)
    eager_loss, eager_measures = minibatch_loss(config, policy, *arguments)
    # Dynamic: the other tests here compile the same functions at other sizes.
    compiled_loss, compiled_measures = graphs("dynamic").minibatch_loss(config, twin, *arguments)
    assert torch.allclose(eager_loss, compiled_loss, rtol=1e-4, atol=1e-5)
    for name in eager_measures:
        assert torch.allclose(
            eager_measures[name], compiled_measures[name], rtol=1e-4, atol=1e-5
        ), name
    eager_loss.backward()
    compiled_loss.backward()
    for (name, eager), compiled_parameter in zip(
        policy.named_parameters(), twin.parameters(), strict=True
    ):
        if eager.grad is None:
            assert compiled_parameter.grad is None or not compiled_parameter.grad.any(), name
            continue
        assert torch.allclose(eager.grad, compiled_parameter.grad, rtol=1e-3, atol=1e-5), name


def test_the_league_tick_compiles_fullgraph_so_it_never_reads_the_host():
    """The capacity routing (slot_actions, one call per slot) and the tick's second half
    (advance: the tallies and the capacity draws at re-deal) trace as single graphs: no
    graph break, so no host read in the tick path; and the one routing graph serves every
    slot and every table width (no recompile when W or K changes)."""
    torch._dynamo.reset()
    torch.manual_seed(0)
    config = LearnConfig(team_size_minimum=2, team_size_maximum=2)
    game_config = game.GameConfig(training_spawn=True, clock_ticks=3)
    policy = model.Policy(model.ModelConfig())
    league = League(config)
    opponent = Opponent(scripted.ScriptedConfig(), league)
    state, carry = start(config, game_config, policy, opponent, 8, None)
    add_snapshots(league, [policy, model.Policy(model.ModelConfig())])
    chosen = torch.tensor([0, EMA, FIRST_SNAPSHOT, FIRST_SNAPSHOT + 1])
    carry = dataclasses.replace(carry, member=chosen, slot=league.member_slot[chosen])
    league.prepare(4, carry.slot)
    slots = state.exists.shape[1]
    deals = draw_next_episodes(config, game_config, policy, opponent, 8, slots, None, "cpu")
    carry = dataclasses.replace(carry, next_episodes=deals)
    graphs("dynamic")  # duck sizing off, as in a dynamic compile
    routing = torch.compile(slot_actions, fullgraph=True, dynamic=True)
    second_half = torch.compile(advance, fullgraph=True, dynamic=True)
    for tick in range(4):  # a game ends on the third tick: re-deals draw members within capacity
        decision = act(config, game_config, policy, opponent, state, carry, None)
        league_move = league_actions(
            config, opponent, decision.views, carry, decision.scripted.action, None, routing
        )
        state, carry, _ = second_half(
            config,
            game_config,
            policy,
            opponent,
            state,
            carry,
            decision,
            league_move,
            None,
        )
    rows = torch.bincount(carry.member.clamp(min=0), minlength=league.members)
    assert (rows <= league.capacity).all()
    # A third snapshot playing every league game: one more slot (K) and a wider table (W,
    # the slot's 4 rows past the capacity); the graph is reused.
    compiles = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    width = league.table.positions.shape[0]
    add_snapshots(league, [model.Policy(model.ModelConfig())])
    chosen = torch.full((4,), FIRST_SNAPSHOT + 2)
    carry = dataclasses.replace(carry, member=chosen, slot=league.member_slot[chosen])
    league.prepare(4, carry.slot)
    assert len(league.table.slots) == 4 and league.table.positions.shape[0] == 4 != width
    decision = act(config, game_config, policy, opponent, state, carry, None)
    league_actions(config, opponent, decision.views, carry, decision.scripted.action, None, routing)
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == compiles


def test_the_first_league_networks_do_not_recompile_the_tick():
    """Before the league has a network (the BC stage) `advance` takes the scripted rows; after
    (RL), the routing graph's output. Both are tensors of their own (rollout.league_rows), so
    neither `advance` nor the routing graph's later slots compile a second graph (a view's
    base is guarded: 'NoneType' object has no attribute 'stride')."""
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()  # earlier tests' graphs
    torch.manual_seed(0)
    config = LearnConfig(team_size_minimum=2, team_size_maximum=2)
    game_config = game.GameConfig(training_spawn=True)
    policy = model.Policy(model.ModelConfig())
    league = League(config)
    opponent = Opponent(scripted.ScriptedConfig(), league)
    state, carry = start(config, game_config, policy, opponent, 8, None)
    slots = state.exists.shape[1]
    graphs("dynamic")  # duck sizing off, as in a dynamic compile
    routing = torch.compile(slot_actions, fullgraph=True, dynamic=True)
    second_half = torch.compile(advance, fullgraph=True, dynamic=True)

    def tick(state, carry, row):
        deals = draw_next_episodes(config, game_config, policy, opponent, 8, slots, None, "cpu")
        carry = dataclasses.replace(carry, next_episodes=deals)
        decision = act(config, game_config, policy, opponent, state, carry, None)
        league_move = league_actions(
            config, opponent, decision.views, carry, decision.scripted.action, None, routing
        )
        state, carry, _ = second_half(
            config,
            game_config,
            policy,
            opponent,
            state,
            carry,
            decision,
            league_move,
            None,
        )
        return state, carry

    league.prepare(4, carry.slot)
    assert league.table is None
    state, carry = tick(state, carry, 4)  # no network: the scripted rows
    add_snapshots(league, [policy, model.Policy(model.ModelConfig())])
    chosen = torch.tensor([0, EMA, FIRST_SNAPSHOT, FIRST_SNAPSHOT + 1])
    carry = dataclasses.replace(carry, member=chosen, slot=league.member_slot[chosen])
    league.prepare(4, carry.slot)
    assert len(league.table.slots) == 3
    for row in (5, 6):
        state, carry = tick(state, carry, row)
    stats = torch._dynamo.utils.counters["stats"]
    assert stats["unique_graphs"] == 2, dict(stats)  # advance once, the routing once
