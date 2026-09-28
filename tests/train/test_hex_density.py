"""The hex density target: grid layout, and exactness against a naive reference."""

import math

import pytest
import torch

from boost_and_broadside.config.core import EnvConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.profiles import PROFILES
from boost_and_broadside.train.rl.features import PRESENCE_RADIUS
from boost_and_broadside.train.rl.hex_density import (
    HEX_DENSITY_RINGS,
    HexDensityTarget,
    hex_cell_count,
    hex_cell_offsets,
    hex_density,
)

SMALL_RINGS = 2
SMALL_WORLD = (4000.0, 3000.0)


def _naive_density(
    ship_xy: list[list[tuple[float, float]]],
    ship_team: list[list[int]],
    ship_alive: list[list[bool]],
    observer_team: list[int],
    cell_xy: list[list[tuple[float, float]]],
    world_size: tuple[float, float],
    radius: float,
) -> torch.Tensor:
    """One env, one cell, one ship at a time, in float64."""
    width, height = world_size
    rows = []
    for env, cells in enumerate(cell_xy):
        ally, enemy = [], []
        for cell_x, cell_y in cells:
            sums = [0.0, 0.0]
            for (x, y), team, alive in zip(ship_xy[env], ship_team[env], ship_alive[env]):
                if not alive:
                    continue
                dx = (cell_x - x + width / 2.0) % width - width / 2.0
                dy = (cell_y - y + height / 2.0) % height - height / 2.0
                weight = math.exp(-(dx * dx + dy * dy) / (2.0 * radius * radius))
                sums[0 if team == observer_team[env] else 1] += weight
            ally.append(math.log1p(sums[0]))
            enemy.append(math.log1p(sums[1]))
        rows.append(ally + enemy)
    return torch.tensor(rows, dtype=torch.float64)


class TestGrid:
    @pytest.mark.parametrize("rings", [1, 2, HEX_DENSITY_RINGS])
    def test_cell_count_is_the_hexagonal_number(self, rings):
        assert hex_cell_offsets(rings).shape == (hex_cell_count(rings), 2)

    def test_production_grid_is_in_the_planned_range(self):
        assert 271 <= hex_cell_count(HEX_DENSITY_RINGS) <= 331

    def test_cells_are_distinct(self):
        cells = hex_cell_offsets(HEX_DENSITY_RINGS)
        gaps = torch.cdist(cells, cells) + torch.eye(cells.shape[0]) * 10.0
        assert gaps.min() > 0.5 / HEX_DENSITY_RINGS

    def test_outer_corners_sit_on_the_playable_boundary(self):
        radii = hex_cell_offsets(HEX_DENSITY_RINGS).norm(dim=-1)
        assert radii.max().item() == pytest.approx(1.0, abs=1e-6)

    def test_ordering_is_ring_by_ring_from_the_positive_x_axis(self):
        cells = hex_cell_offsets(SMALL_RINGS) * SMALL_RINGS
        expected_first_ring = [
            (math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)) for k in range(6)
        ]
        assert cells[0].tolist() == [0.0, 0.0]
        assert torch.allclose(cells[1:7], torch.tensor(expected_first_ring), atol=1e-6)
        assert cells[7].tolist() == pytest.approx([2.0, 0.0])


class TestExactness:
    def test_matches_the_naive_reference_across_the_seam(self):
        generator = torch.Generator().manual_seed(3)
        num_envs, num_ships = 3, 6
        width, height = SMALL_WORLD
        ship_x = torch.rand(num_envs, num_ships, generator=generator) * width
        ship_y = torch.rand(num_envs, num_ships, generator=generator) * height
        ship_team = torch.tensor([[0, 0, 0, 1, 1, 1]] * num_envs)
        alive = torch.rand(num_envs, num_ships, generator=generator) > 0.3
        observer = torch.tensor([0, 1, 0])
        # Centres near the seam so minimum-image displacement is exercised.
        center = torch.tensor([[50.0, 40.0], [3950.0, 2980.0], [2000.0, 1500.0]])
        cells = center.unsqueeze(1) + 1200.0 * hex_cell_offsets(SMALL_RINGS)  # (B, C, 2)

        got = hex_density(
            ship_x,
            ship_y,
            ship_team == observer.unsqueeze(1),
            alive,
            cells[..., 0],
            cells[..., 1],
            SMALL_WORLD,
        )
        expected = _naive_density(
            torch.stack((ship_x, ship_y), dim=-1).tolist(),
            ship_team.tolist(),
            alive.tolist(),
            observer.tolist(),
            cells.tolist(),
            SMALL_WORLD,
            PRESENCE_RADIUS,
        )
        assert torch.allclose(got.double(), expected, atol=1e-5)

    def test_matches_the_naive_reference_on_a_frontline_state(self):
        profile = PROFILES["rl"]
        ship_config = profile.ship_config
        config = EnvConfig(
            num_ships=4,
            num_fields=2,
            max_bullets=profile.max_bullets,
            max_episode_steps=profile.max_episode_steps,
            frontline=profile.frontline,
            vision_range=profile.vision_range,
        )
        env = TensorEnv(2, ship_config, config, "cpu")
        env.reset(seed=11)
        state = env.state
        observer = torch.tensor([1, 0])
        target = HexDensityTarget(ship_config.world_size, "cpu", None, rings=SMALL_RINGS)

        got = target(state, observer)
        center = torch.stack((state.map_center.real, state.map_center.imag), dim=-1)
        cells = center.unsqueeze(1) + state.playable_boundary_radius[:, None, None] * (
            target.unit_cells
        )
        expected = _naive_density(
            torch.stack((state.ship_pos.real, state.ship_pos.imag), dim=-1).tolist(),
            state.ship_team_id.tolist(),
            state.ship_alive.tolist(),
            observer.tolist(),
            cells.tolist(),
            ship_config.world_size,
            PRESENCE_RADIUS,
        )
        assert torch.allclose(got.double(), expected, atol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_compiled_target_equals_eager(self):
        profile = PROFILES["rl"]
        config = EnvConfig(
            num_ships=10,
            num_fields=4,
            max_bullets=profile.max_bullets,
            max_episode_steps=profile.max_episode_steps,
            frontline=profile.frontline,
            vision_range=profile.vision_range,
        )
        env = TensorEnv(8, profile.ship_config, config, "cuda")
        env.reset(seed=2)
        observer = torch.zeros(8, dtype=torch.int32, device="cuda")
        world = profile.ship_config.world_size
        eager = HexDensityTarget(world, "cuda", None)(env.state, observer)
        compiled = HexDensityTarget(world, "cuda", "default")(env.state, observer)
        assert torch.allclose(eager, compiled, atol=1e-5)
