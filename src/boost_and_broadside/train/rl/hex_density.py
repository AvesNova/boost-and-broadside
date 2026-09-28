"""Privileged ally/enemy density on a fixed hex grid over the playable circle.

The target the global token will predict (Phase 7 attaches the loss): for every
cell of a fixed hexagonal grid laid over the Frontline playable circle, how
crowded that point is with the observer's allies and with its enemies, *now*.
It is the ``local_presence`` quantity evaluated at fixed map points instead of
at ships -- the same Gaussian kernel over toroidal distance, the same ``log1p``
compression -- so "crowded" means the same thing in both places.

Targets read privileged truth (every living ship, seen or not); nothing here
ever enters policy input.

Grid. ``rings`` concentric hexagonal rings around the centre cell, so the cell
count is ``1 + 3 * rings * (rings + 1)`` whatever the map size. Cell centres are
stored in units of the playable radius and scaled per environment, so a 50v50
map carries the same cells as a literal zoom of 5v5. The outer ring's corner
cells sit on the playable boundary; its edge midpoints sit at ``sqrt(3)/2`` of
it. Ordering is ring by ring from the centre; within a ring it starts on the
+x axis and walks counter-clockwise. That ordering is part of the target's
meaning and must not change once a head is trained on it.
"""

from __future__ import annotations

import math

import torch

from boost_and_broadside.env.state import TensorState
from boost_and_broadside.train.rl.features import PRESENCE_RADIUS, PRESENCE_SCALE

#: Rings around the centre cell. Ten rings is 331 cells; at the 5v5 playable
#: radius of 2600 px that is a 260 px spacing, about half the 500 px kernel
#: radius, so the field is sampled finer than it varies.
HEX_DENSITY_RINGS = 10

# Axial hex steps, in the order that walks one ring counter-clockwise when the
# walk starts on the +x axis at axial (ring, 0).
_RING_WALK = ((-1, 1), (-1, 0), (0, -1), (1, -1), (1, 0), (0, 1))


def hex_cell_count(rings: int) -> int:
    """Cells in a hexagon of ``rings`` rings around one centre cell."""

    return 1 + 3 * rings * (rings + 1)


def hex_cell_offsets(rings: int) -> torch.Tensor:
    """Cell centres in units of the playable radius, in the grid's fixed order.

    Args:
        rings: Rings around the centre cell. Must be positive.

    Returns:
        (C, 2) float32 x/y offsets from the map centre. Corner cells of the
        outermost ring lie at distance exactly 1.

    Raises:
        ValueError: If ``rings`` is not positive.
    """

    if rings < 1:
        raise ValueError(f"hex density grid needs at least one ring, got {rings}")
    axial = [(0, 0)]
    for ring in range(1, rings + 1):
        q, r = ring, 0
        for step_q, step_r in _RING_WALK:
            for _ in range(ring):
                q, r = q + step_q, r + step_r
                axial.append((q, r))
    # The walk ends back on its starting corner; rotate so each ring starts there.
    offsets = []
    start = 1
    for ring in range(1, rings + 1):
        ring_cells = axial[start : start + 6 * ring]
        offsets.extend([ring_cells[-1], *ring_cells[:-1]])
        start += 6 * ring
    cells = [(0, 0), *offsets]
    xy = torch.tensor(
        [(q + r / 2.0, r * math.sqrt(3.0) / 2.0) for q, r in cells], dtype=torch.float32
    )  # (C, 2)
    return xy / rings


def hex_density(
    ship_x: torch.Tensor,
    ship_y: torch.Tensor,
    is_ally: torch.Tensor,
    contributes: torch.Tensor,
    cell_x: torch.Tensor,
    cell_y: torch.Tensor,
    world_size: tuple[float, float],
    radius: float = PRESENCE_RADIUS,
    scale: float = PRESENCE_SCALE,
) -> torch.Tensor:
    """Ally and enemy density at every cell, exactly, over every contributing ship.

        density[team, c] = log1p( sum_j [team_j == team] exp(-|d_jc|^2 / (2 r^2)) ) / scale

    x and y are carried as separate tensors so nothing ever holds a
    sample x ship x cell x coordinate intermediate; the widest tensor is
    sample x ship x cell, and a compiled caller fuses even that into the sum.

    Args:
        ship_x, ship_y: (B, N) world position.
        is_ally:        (B, N) bool, the ship is on the observer's team.
        contributes:    (B, N) bool, the ship counts at all (alive).
        cell_x, cell_y: (B, C) world position of every cell.
        world_size:     (width, height) of the toroid.
        radius:         Kernel radius in pixels.
        scale:          Divisor applied after ``log1p``.

    Returns:
        (B, 2 * C) -- every cell's ally density, then every cell's enemy density.
    """

    width, height = world_size
    delta_x = cell_x.unsqueeze(1) - ship_x.unsqueeze(2)  # (B, N, C)
    delta_y = cell_y.unsqueeze(1) - ship_y.unsqueeze(2)  # (B, N, C)
    # Minimum image on the torus, matching local_presence.
    delta_x = (delta_x + width / 2.0) % width - width / 2.0
    delta_y = (delta_y + height / 2.0) % height - height / 2.0
    weight = torch.exp(-(delta_x * delta_x + delta_y * delta_y) / (2.0 * radius * radius))
    weight = weight * contributes.unsqueeze(2)  # (B, N, C)
    ally = (weight * is_ally.unsqueeze(2)).sum(dim=1)  # (B, C)
    enemy = (weight * ~is_ally.unsqueeze(2)).sum(dim=1)  # (B, C)
    return torch.log1p(torch.cat((ally, enemy), dim=-1)) / scale  # (B, 2C)


def hex_density_from_state(
    state: TensorState,
    observer_team: torch.Tensor,
    unit_cells: torch.Tensor,
    world_size: tuple[float, float],
    radius: float = PRESENCE_RADIUS,
    scale: float = PRESENCE_SCALE,
) -> torch.Tensor:
    """The privileged density target for one observer per environment.

    Every living ship contributes, visible or not. The grid is centred on
    ``state.map_center`` and scaled by ``state.playable_boundary_radius``, both
    zero in the unbounded combat arena, where every cell collapses onto the
    origin; the target is only meaningful on Frontline.

    Args:
        state:         Environment truth.
        observer_team: (B,) team whose allies fill the first half.
        unit_cells:    (C, 2) from :func:`hex_cell_offsets`, on ``state.device``.
        world_size:    (width, height) of the toroid.
        radius:        Kernel radius in pixels.
        scale:         Divisor applied after ``log1p``.

    Returns:
        (B, 2 * C) -- see :func:`hex_density`.
    """

    playable = state.playable_boundary_radius.unsqueeze(1)  # (B, 1)
    cell_x = state.map_center.real.unsqueeze(1) + playable * unit_cells[:, 0]  # (B, C)
    cell_y = state.map_center.imag.unsqueeze(1) + playable * unit_cells[:, 1]  # (B, C)
    is_ally = state.ship_team_id == observer_team.unsqueeze(1)  # (B, N)
    return hex_density(
        state.ship_pos.real,
        state.ship_pos.imag,
        is_ally,
        state.ship_alive,
        cell_x,
        cell_y,
        world_size,
        radius=radius,
        scale=scale,
    )


class HexDensityTarget:
    """The density target with its grid resident on device, fused when asked.

    Args:
        world_size:   (width, height) of the toroid.
        device:       Where the grid and every target live.
        compile_mode: ``torch.compile`` mode, or None for eager.
        rings:        Rings around the centre cell.
    """

    def __init__(
        self,
        world_size: tuple[float, float],
        device: torch.device | str,
        compile_mode: str | None,
        rings: int = HEX_DENSITY_RINGS,
    ):
        self.world_size = (float(world_size[0]), float(world_size[1]))
        self.unit_cells = hex_cell_offsets(rings).to(device)  # (C, 2)
        self.num_cells = self.unit_cells.shape[0]
        self._build = (
            hex_density_from_state
            if compile_mode is None
            else torch.compile(hex_density_from_state, mode=compile_mode, dynamic=False)
        )

    def __call__(self, state: TensorState, observer_team: torch.Tensor) -> torch.Tensor:
        """(B, 2 * C) density target for ``observer_team`` in each environment."""

        return self._build(state, observer_team, self.unit_cells, self.world_size)
