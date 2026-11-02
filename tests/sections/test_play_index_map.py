"""The index grid (`bnb.play.index_map`): the game's `medium` at every texel, built whole
or a few tiles at a time."""

import math

import numpy as np
import pytest
import torch

from bnb.contracts import GameConfig
from bnb.game import api
from bnb.game.modes.rules import PRESETS
from bnb.game.physics.medium import medium
from bnb.play.index_map import TILE, TILES_PER_CALL, IndexBuilder, build
from bnb.play.snapshot import snapshot

# The code under test draws from torch's default generator (model initialisation, the
# rollout, the scripted opponents): each test runs with it seeded and restored.
pytestmark = pytest.mark.usefixtures("global_generator")


@pytest.fixture(autouse=True, scope="module")
def one_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _map(config: GameConfig, seed: int):
    torch.manual_seed(seed)
    rules, types = api.rules_of(config), api.ship_types()
    state = api.reset(rules, types, api.uniforms(rules, 1))
    state, observation, _ = api.perceive(rules, state)
    return state, snapshot(rules, types, state, observation)


@pytest.mark.parametrize("config", [PRESETS["frontline"], GameConfig(6, 40, 4000.0)])
def test_the_grid_is_medium_at_its_texels(config):
    state, frame = _map(config, seed=7)
    grid, values = build(frame.fields, frame.arena_radius)
    assert values.shape == (grid.size, grid.size) and grid.size % TILE == 0
    assert -grid.origin >= frame.arena_radius  # the grid covers the arena
    generator = torch.Generator().manual_seed(1)
    column, row = torch.randint(0, grid.size, (2, 40_000), generator=generator)
    inside = torch.from_numpy(values != 0.0).nonzero()
    row = torch.cat([row, inside[:: max(len(inside) // 10_000, 1), 0]])
    column = torch.cat([column, inside[:: max(len(inside) // 10_000, 1), 1]])
    x = (grid.origin + (column + 0.5) * grid.texel).float()
    y = (grid.origin + (row + 0.5) * grid.texel).float()
    want = medium(state.fields, x[None], y[None])[0][0]
    assert torch.allclose(torch.from_numpy(values)[row, column], want, atol=1e-5)
    # Texels the builder skipped hold exactly 0, as medium gives there.
    assert (want[torch.from_numpy(values)[row, column] == 0.0] == 0.0).all()


def test_a_budgeted_build_matches_the_whole_build():
    _, frame = _map(PRESETS["deathmatch"], seed=3)
    grid, whole = build(frame.fields, frame.arena_radius)
    builder = IndexBuilder(frame.fields, frame.arena_radius)
    parts = np.zeros_like(whole)
    calls = 0
    while not builder.done:
        for update in builder.run(budget=0.0):  # one batch per call
            parts[update.row : update.row + TILE, update.column : update.column + TILE] = (
                update.values
            )
        calls += 1
    assert calls == math.ceil(len(builder.tiles) / TILES_PER_CALL) and calls > 1
    assert np.array_equal(parts, whole)
