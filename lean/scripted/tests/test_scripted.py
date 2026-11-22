"""Invariants of the scripted opponent and the chaser, and one match between them."""

import math

import pytest
import torch

from lean.contract import BULLET_SLOTS, FIRES, POWERS, STICKS, State
from lean.scripted import (
    ScriptedConfig,
    act,
    chaser_act,
    draw_uniforms,
    fresh_memory,
    marginals,
    personas,
)


def random_state(envs: int, slots: int, seed: int) -> State:
    """A state with random ships inside a 3 km arena, two teams by slot parity, and a
    few dead and non-existent rows; no game package needed."""
    generator = torch.Generator().manual_seed(seed)

    def uniform(low: float, high: float) -> torch.Tensor:
        return low + (high - low) * torch.rand((envs, slots), generator=generator)

    heading = uniform(-math.pi, math.pi)
    angle = uniform(-0.2, 0.2)
    exists = uniform(0.0, 1.0) > 0.15
    bullets = (envs, slots, BULLET_SLOTS)
    return State(
        x=uniform(-2000.0, 2000.0),
        y=uniform(-2000.0, 2000.0),
        heading_cos=heading.cos(),
        heading_sin=heading.sin(),
        speed=uniform(100.0, 440.0),
        angle=angle,
        angle_cos=angle.cos(),
        angle_sin=angle.sin(),
        nose_rate=uniform(-0.5, 0.5),
        energy=uniform(0.0, 497563.0),
        shield=uniform(0.0, 100.0),
        shield_wait=torch.zeros(envs, slots),
        cooldown=(uniform(0.0, 3.0)).long(),
        next_slot=torch.zeros(envs, slots, dtype=torch.long),
        alive=exists & (uniform(0.0, 1.0) > 0.2),
        exists=exists,
        team=(torch.arange(slots) % 2).expand(envs, slots).clone(),
        type=torch.zeros(envs, slots, dtype=torch.long),
        constants={},
        bullet_x=torch.zeros(bullets),
        bullet_y=torch.zeros(bullets),
        bullet_dir_x=torch.zeros(bullets),
        bullet_dir_y=torch.zeros(bullets),
        bullet_speed=torch.zeros(bullets),
        bullet_moves_left=torch.zeros(bullets, dtype=torch.long),
        tick=torch.zeros(envs, dtype=torch.long),
        fired=torch.zeros(bullets[:2], dtype=torch.bool),
        bullet_real=torch.zeros(bullets, dtype=torch.bool),
        bullet_fired_tick=torch.zeros(bullets, dtype=torch.long),
    )


@pytest.mark.parametrize("spread", [0.0, 1.0])
def test_actions_in_range_and_zero_for_uncontrolled_rows(spread):
    """Catches an out-of-range index or an action leaking onto a dead, absent or enemy row."""
    config = ScriptedConfig()
    state = random_state(envs=32, slots=10, seed=1)
    generator = torch.Generator().manual_seed(2)
    persona = personas(config, 32, 10, generator, "cpu", spread)
    memory = fresh_memory(32, 10, "cpu")
    limits = torch.tensor([STICKS, POWERS, FIRES])
    for team in (0, 1):
        controlled = state.exists & state.alive & (state.team == team)
        candidates = (
            act(config, state, team, persona, draw_uniforms(32, 10, generator, "cpu")),
            act(config, state, team, persona, draw_uniforms(32, 10, generator, "cpu"), memory),
            chaser_act(config, state, team),
        )
        for action in candidates:
            assert action.shape == (32, 10, 3) and action.dtype == torch.long
            assert (action >= 0).all() and (action < limits).all()
            assert (action[~controlled] == 0).all()
        assert candidates[0][controlled].any()


def test_marginals_are_distributions():
    """Catches a gate or floor that lets a head's probabilities leave [0, 1] or not sum to 1."""
    config = ScriptedConfig()
    state = random_state(envs=32, slots=10, seed=3)
    persona = personas(config, 32, 10, torch.Generator().manual_seed(4), "cpu", 1.0)
    parts = marginals(config, state, 0, persona)
    rows = parts.controlled
    for head in (parts.stick[rows], parts.power[rows]):
        assert torch.isfinite(head).all() and (head >= 0).all()
        assert torch.allclose(head.sum(-1), torch.ones(head.shape[0]), atol=1e-5)
    assert ((parts.fire[rows] >= 0) & (parts.fire[rows] <= 1)).all()


def test_spread_zero_is_one_fixed_setting():
    """Catches hidden randomness or state in the yardstick: equal inputs, equal outputs,
    and the same persona for every ship whatever the generator."""
    config = ScriptedConfig()
    state = random_state(envs=16, slots=10, seed=5)
    first = personas(config, 16, 10, torch.Generator().manual_seed(6), "cpu", 0.0)
    second = personas(config, 16, 10, torch.Generator().manual_seed(7), "cpu", 0.0)
    for name in first:
        assert torch.equal(first[name], second[name])
        assert (first[name] == first[name][0, 0]).all()
    uniforms = draw_uniforms(16, 10, torch.Generator().manual_seed(8), "cpu")
    assert torch.equal(act(config, state, 0, first, uniforms), act(config, state, 0, second, uniforms))


def test_turns_toward_a_target_on_its_left():
    """Catches a flipped side: a target at +90 degrees must get the left-turning stick."""
    config = ScriptedConfig()
    state = random_state(envs=1, slots=2, seed=9)
    state.exists[:] = True
    state.alive[:] = True
    state.x[:] = torch.tensor([[0.0, 0.0]])
    state.y[:] = torch.tensor([[0.0, 400.0]])
    state.heading_cos[:], state.heading_sin[:] = 1.0, 0.0
    state.angle[:], state.angle_cos[:], state.angle_sin[:] = 0.0, 1.0, 0.0
    state.speed[:] = 300.0
    parts = marginals(config, state, 0, personas(config, 1, 2, None, "cpu", 0.0))
    left_indices = [2, 4, 6] if config.left_is_even else [1, 3, 5]
    assert parts.stick[0, 0, left_indices].sum() > 0.9
    assert chaser_act(config, state, 0)[0, 0, 0].item() == left_indices[-1]


def test_opponent_beats_chaser_five_a_side():
    """Catches an opponent that does not fight: over a few dozen 5v5 games it must win
    clearly more than half against the chaser, seats alternated."""
    game = pytest.importorskip("lean.game")
    for name in ("GameConfig", "reset", "step", "draw_noise"):
        if not hasattr(game, name):
            pytest.skip(f"lean.game.{name} is not written yet")
    envs, slots, horizon_ticks = 48, 10, 3000
    generator = torch.Generator().manual_seed(11)
    game_config = game.GameConfig()
    config = ScriptedConfig(arena_radius=game_config.arena_radius)
    team_sizes = torch.full((envs, 2), 5, dtype=torch.long)
    state = game.reset(game_config, envs, team_sizes, generator, "cpu")
    persona = personas(config, envs, slots, generator, "cpu", 0.0)
    # The opponent plays team 0 in the first half of the environments and team 1 in the rest.
    opponent_team = (torch.arange(envs) >= envs // 2).long()
    is_opponent = state.team == opponent_team[:, None]
    result = torch.full((envs,), -1, dtype=torch.long)
    for _ in range(horizon_ticks):
        uniforms = draw_uniforms(envs, slots, generator, "cpu")
        scripted = act(config, state, 0, persona, uniforms) + act(config, state, 1, persona, uniforms)
        chaser = chaser_act(config, state, 0) + chaser_act(config, state, 1)
        action = torch.where(is_opponent[..., None], scripted, chaser)
        noise = game.draw_noise(game_config, envs, slots, generator, "cpu")
        state, out = game.step(game_config, state, action, noise)
        result = torch.where((result < 0) & out.done, out.result, result)
        if (result >= 0).all():
            break
    wins = (result == opponent_team).sum().item()
    losses = (result == 1 - opponent_team).sum().item()
    ended = (result >= 0).sum().item()
    print(f"opponent against chaser: {wins} wins, {losses} losses, {ended} ended of {envs}")
    assert ended >= envs // 2
    assert wins > 0.65 * ended
