"""The compiled training path (lean.learn.compiled) on the CPU: it traces, and the compiled
loss equals the eager one. Slow: every graph compiles."""

import copy
import dataclasses

import pytest
import torch

from lean import game, model, scripted
from lean.learn import (
    League,
    LearnConfig,
    Opponent,
    WeightAverages,
    collect,
    make_optimiser,
    start,
    update,
)
from lean.learn.compiled import graphs
from lean.learn.pool import DRAINING, EMA, FIRST_SNAPSHOT, SCRIPTED
from lean.learn.rollout import (
    act,
    advance,
    draw_next_episodes,
    league_actions,
    merge_league,
    self_play_count,
    slot_actions,
)
from lean.learn.tests.setting import add_snapshots, place

pytestmark = pytest.mark.slow


def test_compiled_collect_and_update_run_with_the_real_packages():
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    config = LearnConfig(
        team_size_minimum=2,
        team_size_maximum=2,
        minibatches=2,
        epochs=1,
        league_capacity_factor=12.0,  # C = L: members placed by hand below
    )
    game_config = game.GameConfig()
    policy = model.Policy(model.ModelConfig())
    league = League(config)
    opponent = Opponent(scripted.ScriptedConfig(), league)
    state, carry = start(config, game_config, policy, opponent, 8, generator)
    add_snapshots(league, [policy, model.Policy(model.ModelConfig())], carry.member, carry.slot)
    # A persona and networks (the EMA, snapshots): the routing and the per-row scripted call.
    carry = place(league, carry, [EMA, FIRST_SNAPSHOT, league.first_persona, FIRST_SNAPSHOT])
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
    """The routing (slot_actions, one call per network slot) and the tick's halves (act with
    the per-row scripted parameters; advance with the tallies and level 2's slot draws at
    re-deal) trace as single graphs: no graph break, so no host read in the tick path."""
    torch._dynamo.reset()
    torch.manual_seed(0)
    config = LearnConfig(team_size_minimum=2, team_size_maximum=2, league_capacity_factor=7.5)
    game_config = game.GameConfig(training_spawn=True, clock_ticks=3)
    policy = model.Policy(model.ModelConfig())
    league = League(config)
    opponent = Opponent(scripted.ScriptedConfig(), league)
    state, carry = start(config, game_config, policy, opponent, 8, None)
    add_snapshots(league, [policy, model.Policy(model.ModelConfig())], carry.member, carry.slot)
    carry = place(league, carry, [league.first_persona, EMA, FIRST_SNAPSHOT, FIRST_SNAPSHOT + 1])
    league.prepare(4, carry.slot)
    assert league.width == 3  # ceil(7.5 * 4 / 12)
    slots = state.exists.shape[1]
    deals = draw_next_episodes(config, game_config, policy, opponent, 8, slots, None, "cpu")
    carry = dataclasses.replace(carry, next_episodes=deals)
    graphs("dynamic")  # duck sizing off, as in a dynamic compile
    first_half = torch.compile(act, fullgraph=True, dynamic=True)
    routing = torch.compile(slot_actions, fullgraph=True, dynamic=True)
    second_half = torch.compile(advance, fullgraph=True, dynamic=True)
    for _ in range(4):  # a game ends on the third tick: re-deals draw slots within capacity
        decision = first_half(config, game_config, policy, opponent, state, carry, None)
        league_move = league_actions(
            config, opponent, decision.views, carry, decision.scripted.action, None, routing
        )
        state, carry, _ = second_half(
            config, game_config, policy, opponent, state, carry, decision, league_move, None
        )
    rows = torch.bincount(carry.slot, minlength=league.slot_count + 1)
    assert (rows[: league.slot_count] <= league.capacity).all() and rows[-1] == 0


def test_the_league_tick_is_static_through_loads_unloads_and_duplicates():
    """Under the static compile (training's: a recompile raises), the tick's graphs (act,
    advance, the routing, the merge) compile once at RL's start, where the league opens with
    the EMA and the personas, and serve every later table unchanged: snapshots joining (loads
    into free slots), targets moving so that slots unload (drain) and members hold several
    slots (duplicates), a snapshot dropped by retention (its slots drain), the network slots
    run growing and shrinking."""
    torch._dynamo.reset()
    torch.manual_seed(0)
    config = LearnConfig(
        team_size_minimum=2,
        team_size_maximum=2,
        league_capacity_factor=7.5,
        snapshot_members=2,
        recent_snapshots=2,
    )
    game_config = game.GameConfig(training_spawn=True, clock_ticks=6)
    policy = model.Policy(model.ModelConfig())
    averages = WeightAverages(policy, config)
    league = League(config)
    league.open(policy, averages)
    opponent = Opponent(scripted.ScriptedConfig(), league)
    state, carry = start(config, game_config, policy, opponent, 16, None)
    from lean.contract import compile_graph

    # Its own graphs: graphs() is per process.
    first_half, second_half = compile_graph(act, "static"), compile_graph(advance, "static")
    routing, merge = compile_graph(slot_actions, "static"), compile_graph(merge_league, "static")
    slots = state.exists.shape[1]
    calls, duplicated, drained = [], False, False

    def ticks(state, carry, count):
        league.prepare(8, carry.slot)
        calls.append(0 if league.table is None else len(league.table.slots))
        deals = draw_next_episodes(config, game_config, policy, opponent, 16, slots, None, "cpu")
        carry = dataclasses.replace(carry, next_episodes=deals)
        for _ in range(count):
            decision = first_half(config, game_config, policy, opponent, state, carry, None)
            league_move = league_actions(
                config,
                opponent,
                decision.views,
                carry,
                decision.scripted.action,
                None,
                routing,
                merge,
            )
            state, carry, _ = second_half(
                config, game_config, policy, opponent, state, carry, decision, league_move, None
            )
        return state, carry

    state, carry = ticks(state, carry, 3)  # RL's first tick: the EMA plays, nothing more joins
    steps = list(league.schedule.points(league.schedule.steps[-1] * 4))[:5]
    for index, step in enumerate(steps):
        win = {SCRIPTED: 0.95, EMA: 0.95} if index % 2 else {FIRST_SNAPSHOT: 0.0}
        for member, rate in win.items():
            league.games[member], league.wins[member] = 50.0, 50.0 * rate
        member = league.after_update(policy, averages, step - 1, step, carry.member, carry.slot)
        carry = dataclasses.replace(carry, member=member)
        duplicated |= max(league.held()) >= 2 and any(
            league.held()[m] >= 2 for m in range(1, league.first_persona)
        )
        drained |= DRAINING in league.slot_state
        state, carry = ticks(state, carry, 3)  # a recompile here would raise
    assert duplicated and drained and league.taken == 5 and league.snapshots == 2
    assert len(set(calls)) > 1  # the network slots run changed from table to table
