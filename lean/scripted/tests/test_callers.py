"""Decision 5.2: every caller carries the scripted agent's memory from call to call and starts
it afresh for each match (collect's teacher and league opponent, evaluation, the sandbox)."""

import torch

from lean import game, model, scripted
from lean.eval import Queue, play
from lean.learn import collect
from lean.learn.tests.setting import small
from lean.sandbox.world import World
from lean.scripted.parameters import vector


class MemoryRecorder:
    """Wraps lean.scripted.decide (install with pytest's monkeypatch) and keeps, per call, the
    state's episode tick, the memory passed in and the memory returned."""

    def __init__(self):
        self.calls = []
        self.decide = scripted.decide

    def __call__(self, config, state, team, persona, uniforms, memory):
        action, new_memory, parts = self.decide(config, state, team, persona, uniforms, memory)
        self.calls.append(
            (
                state.tick.clone(),
                {name: value.clone() for name, value in memory.items()},
                new_memory,
            )
        )
        return action, new_memory, parts

    def check(self, discarded=()) -> int:
        """Each call's memory is the previous call's, except in environments starting a new
        game (episode tick 0), whose kept target is fresh (-1). Calls whose result the caller
        discards (by index) are left out. Returns the fresh starts seen."""
        calls = [call for index, call in enumerate(self.calls) if index not in discarded]
        fresh = 0
        for (_, _, before), (tick, given, _) in zip(calls[:-1], calls[1:], strict=True):
            new_game = tick == 0
            assert torch.equal(given["target"][~new_game], before["target"][~new_game])
            assert torch.equal(given["side"][~new_game], before["side"][~new_game])
            assert (given["target"][new_game] == -1).all()
            fresh += int(new_game.sum())
        return fresh


def test_collect_carries_the_memory_and_renews_it_at_each_deal(monkeypatch):
    """Collect's one scripted call per tick (teacher and league opponent) is handed the memory
    the previous call returned, and a fresh one in an environment re-dealt for a new game."""
    recorder = MemoryRecorder()
    monkeypatch.setattr(scripted, "decide", recorder)
    config, game_config, policy, opponent, state, carry, generator = small(
        8, game_config=game.GameConfig(training_spawn=True, clock_ticks=6)
    )
    for _ in range(2):  # the memory runs on across segments
        rollout = collect(config, game_config, policy, opponent, state, carry, 8, generator)
        state, carry = rollout.state, rollout.carry
    # A segment's 9th call (the bootstrap value's first half) discards its scripted actions.
    assert len(recorder.calls) == 18
    assert recorder.check(discarded={8, 17}) > 0
    assert any(
        (memory["target"] >= 0).any() for _, _, memory in recorder.calls
    )  # targets were kept


def test_evaluation_carries_the_memory_and_renews_it_for_each_game(monkeypatch):
    """Evaluation's scripted opponent: the same rule over games that end and restart."""
    recorder = MemoryRecorder()
    monkeypatch.setattr(scripted, "decide", recorder)
    torch.manual_seed(0)
    games = torch.zeros(24, dtype=torch.long)
    queue = Queue(games, games, games, games + 2, games + 2)  # 24 games on 8 slots: three each
    outcome = play(
        game.GameConfig(clock_ticks=6),
        queue,
        torch.tensor([vector(scripted.ScriptedConfig())]),
        (model.Policy(model.ModelConfig()),),
        envs=8,
        segment=4,
    )
    assert (outcome.result >= 0).all()
    assert len(recorder.calls) >= 18
    assert recorder.check() >= 16  # every slot's second and third game started fresh


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
