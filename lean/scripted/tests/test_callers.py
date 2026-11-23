"""Decision 5.2: every caller carries the scripted agent's memory from call to call and starts
it afresh for each match (collect's teacher and league opponent, evaluation, the sandbox)."""

from types import SimpleNamespace

import torch

from lean import game, scripted
from lean.eval import play_results
from lean.learn import LearnConfig
from lean.learn.collect import Opponent, collect, start
from lean.learn.tests import standins
from lean.sandbox.world import World


def test_collect_carries_the_memory_and_renews_it_at_each_deal():
    """The stand-in's memory counts the calls since its match began, so after a segment every
    environment's count is its episode tick minus one: carried every tick, fresh at a re-deal."""
    generator = torch.Generator().manual_seed(0)
    config = LearnConfig(team_size_minimum=2, team_size_maximum=2)
    policy = standins.TinyPolicy()
    opponent = Opponent(scripted_config=None)
    options = dict(game=standins.game, scripted=standins.scripted)
    state, carry = start(config, standins.GameConfig(), policy, opponent, 8, generator, **options)
    rollout = collect(
        config, standins.GameConfig(), policy, opponent, state, carry, 16, generator, **options
    )
    assert int(rollout.episodes) > 0  # some environments were re-dealt (6-tick episodes)
    memory = rollout.carry.scripted_memory["target"]
    assert torch.equal(memory, (rollout.state.tick[:, None] - 1).expand_as(memory))


def test_evaluation_carries_the_memory_and_renews_it_for_each_game():
    """Each call's memory is the previous call's (renewed for a finished game): the stand-in's
    count equals the episode tick minus one on every call."""
    calls = []

    def act(config, state, team, persona, uniforms, memory):
        calls.append(
            bool(
                torch.equal(memory["target"], (state.tick[:, None] - 1).expand_as(memory["target"]))
            )
        )
        return standins.scripted.act(config, state, team, persona, uniforms, memory)

    recording = SimpleNamespace(**{**vars(standins.scripted), "act": act})
    sizes = torch.full((8, 2), 2, dtype=torch.long)
    play_results(
        standins.GameConfig(),
        standins.TinyPolicy(),
        None,
        sizes,
        torch.Generator().manual_seed(1),
        maximum_ticks=20,
        check_every=100,
        game=standins.game,
        scripted=recording,
    )
    assert len(calls) == 20 and all(calls)


def test_sandbox_flies_the_scripted_opponent_with_its_memory():
    """The sandbox's default opponent is the scripted agent, its memory carried: after a few
    ticks it keeps the player (slot 0) as its target and its head-on side is unchanged."""
    world = World(game.GameConfig(), seed=3)
    assert world.opponent == "scripted"
    side = world.memory["side"].clone()
    for _ in range(5):
        world.advance(0, 0, False)
    assert int(world.memory["target"][0, 1]) == 0
    assert torch.equal(world.memory["side"], side)
    chaser = World(game.GameConfig(), seed=3, opponent="chaser")
    chaser.advance(0, 0, False)
    assert scripted.ScriptedConfig() == chaser.scripted
