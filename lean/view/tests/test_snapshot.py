"""The adaption layer reads lean's state as the game has it (values read from the code)."""

import dataclasses

import numpy as np
import pytest
import torch

from lean import game, scripted
from lean.contract import BULLET_SLOTS
from lean.game import core
from lean.game.tests.values import game_value
from lean.view.index_map import build
from lean.view.match import MatchSettings, record
from lean.view.snapshot import snapshot, to_v2


def played(config: core.GameConfig, sizes=(2, 2), ticks: int = 40, seed: int = 0):
    """A deal and `ticks` scripted ticks: (states, step outputs)."""
    generator = torch.Generator().manual_seed(seed)
    slots = sum(sizes)
    state = game.deal(config, 1, slots, torch.tensor([list(sizes)]), generator, "cpu")
    memory = scripted.fresh_memory(1, slots, generator, "cpu")
    agent = scripted.ScriptedConfig()
    states, outs = [state], [None]
    for _ in range(ticks):
        uniforms = scripted.draw_uniforms(1, slots, generator, "cpu")
        action, memory, _ = scripted.decide(agent, state, None, None, uniforms, memory)
        state, out = game.tick(config, state, action, generator)
        states.append(state)
        outs.append(out)
    return states, outs


def test_ships_read_the_card_and_the_state():
    config = core.GameConfig()
    states, outs = played(config)
    frame = snapshot(config, states[-1], step_out=outs[-1], previous=states[-2])
    ships = frame.ships
    assert len(ships.x) == 4 and list(ships.team) == [0, 0, 1, 1]
    np.testing.assert_allclose(ships.hit_radius, game_value("hit_radius", config))
    np.testing.assert_allclose(ships.vision_range, game_value("vision_range", config))
    np.testing.assert_allclose(ships.x, states[-1].x[0].numpy())
    shield = states[-1].shield[0].numpy() / game_value("max_shield", config)
    np.testing.assert_allclose(ships.shield, shield, rtol=1e-6)
    assert ((ships.energy >= 0) & (ships.energy <= 1)).all()
    assert frame.arena_radius == config.arena_radius
    assert frame.tick == 40 and frame.clock == pytest.approx(40 * config.tick_length)
    assert len(frame.fields.x) == config.field_count
    # A team always sees its own ships.
    for team in (0, 1):
        assert ships.visible[team][ships.team == team].all()
        assert not ships.tracked[team][ships.team == team].any()


def test_bullets_keep_every_slot_and_only_real_ones_are_active():
    config = core.GameConfig()
    states, outs = played(config, ticks=60)
    frame = snapshot(config, states[-1], step_out=outs[-1], previous=states[-2])
    bullets = frame.bullets
    state = states[-1]
    assert len(bullets.x) == 4 * BULLET_SLOTS
    real = ((state.bullet_moves_left[0] > 0) & state.bullet_real[0]).reshape(-1).numpy()
    np.testing.assert_array_equal(bullets.active, real)
    assert (bullets.age[~bullets.active] == 0).all()


def test_fog_off_shows_every_ship_to_both_teams():
    config = dataclasses.replace(core.GameConfig(), fog=False)
    states, _ = played(config, ticks=5)
    frame = snapshot(config, states[-1])
    assert frame.ships.visible.all() and not frame.fogged


def test_v2_snapshot_lines_up():
    config = core.GameConfig()
    states, outs = played(config, ticks=10)
    frame = snapshot(config, states[-1], step_out=outs[-1], previous=states[-2])
    v2 = to_v2(frame)
    assert v2.ships.visible.shape == (2, len(frame.ships.x))
    assert len(v2.bullets.active) == len(frame.bullets.x)
    assert not v2.zones.present.any()
    assert v2.boundary_radius == config.arena_radius


def test_index_grid_is_the_games_medium():
    config = core.GameConfig()
    states, _ = played(config, ticks=0)
    frame = snapshot(config, states[0])
    grid = build(frame.fields, config.arena_radius)
    rows = np.random.default_rng(0).integers(0, grid.size, (2, 200))
    x = grid.origin + (rows[1] + 0.5) * grid.texel
    y = grid.origin + (rows[0] + 0.5) * grid.texel
    expected, _, _ = core.medium(
        states[0].fields,
        torch.tensor(x[None], dtype=torch.float32),
        torch.tensor(y[None], dtype=torch.float32),
    )
    np.testing.assert_allclose(grid.values[rows[0], rows[1]], expected[0].numpy(), atol=1e-6)


def test_a_whole_match_records_its_deaths():
    frames, _ = record(MatchSettings(ships_per_team=(1, 1), seed=0, fields=False))
    deaths = sum(int(frame.events.died.sum()) for frame in frames)
    left = int(frames[-1].ships.alive.sum())
    assert deaths == 2 - left
    assert frames[-1].result >= 0 and all(frame.result == -1 for frame in frames[:-1])
