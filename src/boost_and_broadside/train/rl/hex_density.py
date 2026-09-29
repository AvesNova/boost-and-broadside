"""Privileged ally/enemy ship counts on a fixed hex grid over the playable circle.

The target the global token predicts: for every cell of a fixed hexagonal grid
laid over the Frontline playable circle, how much of an ally and how much of an
enemy sits there, *now*. Each living ship deposits exactly one unit of mass,
split across the three cells whose centres form the triangle containing it, by
barycentric weight. So the field sums to the living ship count per side and a
cell's value is a soft count, not a density.

No kernel and no radius. Three weights that sum to one are the exact 2D analogue
of the "two-hot" encoding of a continuous value on a line, and they are lossless:
a single ship's three weights invert to its exact position inside the triangle.
A Gaussian smoothing, by contrast, has a width that has to be chosen, and it
cost real resolution -- at 500 px one ship lit a fifth of the grid and two ships
did not read as two until 2000 px apart
(``docs/internal/density-audit-sep2026.md``).

Because the field is counts, the head that predicts it is a Poisson intensity:
it emits a log-rate per cell and is trained by Poisson negative log likelihood,
whose gradient in the logit is the bounded ``exp(logit) - count``. That is also
why nothing here is normalized -- a softmax would pin the total mass at one and
throw away the count, which is half the information.

Off the grid. The lattice is treated as infinite: a ship's containing triangle
is found wherever the ship is, and only those of its three vertices that are
cells of the finite grid receive their weight. A ship just outside the boundary
therefore still deposits part of itself on the edge cells it is next to, its
contribution falling continuously to nothing as it leaves -- no clamping, no
special case, and no discontinuity at the rim.

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

#: Rings around the centre cell. Ten rings is 331 cells; at the 5v5 playable
#: radius of 2600 px that is a 260 px spacing, so a ship's position is resolved
#: to the triangle it occupies and, through its barycentric weights, exactly
#: within it.
HEX_DENSITY_RINGS = 10

# Axial hex steps, in the order that walks one ring counter-clockwise when the
# walk starts on the +x axis at axial (ring, 0).
_RING_WALK = ((-1, 1), (-1, 0), (0, -1), (1, -1), (1, 0), (0, 1))


def hex_cell_count(rings: int) -> int:
    """Cells in a hexagon of ``rings`` rings around one centre cell."""

    return 1 + 3 * rings * (rings + 1)


#: Width of one observer's density target, and of the head that predicts it:
#: every cell's ally density followed by every cell's enemy density.
HEX_DENSITY_DIM = 2 * hex_cell_count(HEX_DENSITY_RINGS)


def hex_axial_cells(rings: int) -> list[tuple[int, int]]:
    """Axial ``(q, r)`` of every cell, in the grid's fixed order.

    The single source of the ordering: :func:`hex_cell_offsets` places these in
    the plane and :func:`hex_axial_lookup` indexes them, so a cell's position
    and its column can never disagree.

    Args:
        rings: Rings around the centre cell. Must be positive.

    Returns:
        ``1 + 3 * rings * (rings + 1)`` axial coordinates, centre first, then
        ring by ring; within a ring, starting on the +x axis and walking
        counter-clockwise.

    Raises:
        ValueError: If ``rings`` is not positive.
    """

    if rings < 1:
        raise ValueError(f"hex density grid needs at least one ring, got {rings}")
    walk = [(0, 0)]
    for ring in range(1, rings + 1):
        q, r = ring, 0
        for step_q, step_r in _RING_WALK:
            for _ in range(ring):
                q, r = q + step_q, r + step_r
                walk.append((q, r))
    # The walk ends back on its starting corner; rotate so each ring starts there.
    cells = [(0, 0)]
    start = 1
    for ring in range(1, rings + 1):
        ring_cells = walk[start : start + 6 * ring]
        cells.extend([ring_cells[-1], *ring_cells[:-1]])
        start += 6 * ring
    return cells


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

    cells = hex_axial_cells(rings)
    xy = torch.tensor(
        [(q + r / 2.0, r * math.sqrt(3.0) / 2.0) for q, r in cells], dtype=torch.float32
    )  # (C, 2)
    return xy / rings


def hex_axial_lookup(rings: int) -> torch.Tensor:
    """Dense ``(q, r) -> column`` table, ``-1`` where the lattice has no cell.

    The barycentric target lands on an unbounded lattice and keeps only the
    vertices this finite grid actually has, so the lookup has to answer "not
    here" as readily as it answers a column. Indexed by ``q + pad`` and
    ``r + pad`` with ``pad = rings + 1``: one ring of slack, which is every
    lattice site a triangle touching the grid can reach.

    Args:
        rings: Rings around the centre cell.

    Returns:
        (2 * rings + 3, 2 * rings + 3) int64. Entries are grid columns or -1.
    """

    pad = rings + 1
    table = torch.full((2 * pad + 1, 2 * pad + 1), -1, dtype=torch.int64)
    for column, (q, r) in enumerate(hex_axial_cells(rings)):
        table[q + pad, r + pad] = column
    return table


def barycentric_cells(
    offset_x: torch.Tensor,
    offset_y: torch.Tensor,
    rings: int,
    lookup: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split each point across the three lattice sites of its triangle.

    The hex grid's centres form a triangular lattice, so every point in the
    plane -- on the grid or well off it -- lies in exactly one triangle of three
    mutually adjacent centres, and its barycentric coordinates there are three
    non-negative weights summing to one. They are continuous in position: a
    point crossing a triangle edge arrives with zero weight on the vertex it is
    leaving.

    Args:
        offset_x, offset_y: (..., ) position relative to the map centre, in
            units of the playable radius, minimum-imaged already.
        rings:  Rings around the centre cell.
        lookup: From :func:`hex_axial_lookup`, on the same device.

    Returns:
        ``(column, weight)``, both ``(..., 3)``. A column is ``-1`` where that
        vertex is not a cell of this grid; its weight is still the true
        barycentric weight, and the caller drops it.
    """

    # Cartesian -> fractional axial. One lattice step is 1 / rings, and the two
    # axial basis vectors are (1, 0) and (1/2, sqrt(3)/2) in those units.
    scaled_y = offset_y * rings / (math.sqrt(3.0) / 2.0)
    fractional_r = scaled_y
    fractional_q = offset_x * rings - fractional_r / 2.0

    base_q = torch.floor(fractional_q)
    base_r = torch.floor(fractional_r)
    u = fractional_q - base_q  # in [0, 1)
    v = fractional_r - base_r

    # The rhombus (base .. base + 1) splits along u + v = 1 into a lower
    # triangle at the base corner and an upper one at the far corner. Both
    # assign u to (q+1, r) and v to (q, r+1) on the shared edge, so the weights
    # are continuous across it.
    lower = (u + v) <= 1.0
    zero = torch.zeros_like(u)
    one = torch.ones_like(u)

    corner_q = torch.stack(
        [
            torch.where(lower, zero, one),  # (q, r)   |  (q+1, r)
            torch.where(lower, one, zero),  # (q+1, r) |  (q, r+1)
            torch.where(lower, zero, one),  # (q, r+1) |  (q+1, r+1)
        ],
        dim=-1,
    )
    corner_r = torch.stack(
        [
            torch.where(lower, zero, zero),
            torch.where(lower, zero, one),
            torch.where(lower, one, one),
        ],
        dim=-1,
    )
    weight = torch.stack(
        [
            torch.where(lower, 1.0 - u - v, 1.0 - v),
            torch.where(lower, u, 1.0 - u),
            torch.where(lower, v, u + v - 1.0),
        ],
        dim=-1,
    )  # (..., 3), non-negative and summing to one in both branches

    pad = rings + 1
    size = lookup.shape[0]
    q_index = (base_q.unsqueeze(-1) + corner_q + pad).long()
    r_index = (base_r.unsqueeze(-1) + corner_r + pad).long()
    # A point far off the grid indexes outside the table entirely; clamp so the
    # gather is in bounds and mark it missing rather than letting it wrap.
    inside = (q_index >= 0) & (q_index < size) & (r_index >= 0) & (r_index < size)
    column = lookup[q_index.clamp(0, size - 1), r_index.clamp(0, size - 1)]
    return torch.where(inside, column, torch.full_like(column, -1)), weight


def hex_counts(
    ship_x: torch.Tensor,
    ship_y: torch.Tensor,
    is_ally: torch.Tensor,
    contributes: torch.Tensor,
    centre_x: torch.Tensor,
    centre_y: torch.Tensor,
    playable: torch.Tensor,
    world_size: tuple[float, float],
    rings: int,
    lookup: torch.Tensor,
) -> torch.Tensor:
    """Soft ally and enemy ship counts per cell, exactly.

    Args:
        ship_x, ship_y: (B, N) world position.
        is_ally:        (B, N) bool, the ship is on the observer's team.
        contributes:    (B, N) bool, the ship counts at all (alive).
        centre_x, centre_y: (B, 1) map centre.
        playable:       (B, 1) playable boundary radius, the grid's unit.
        world_size:     (width, height) of the toroid.
        rings:          Rings around the centre cell.
        lookup:         From :func:`hex_axial_lookup`, on the same device.

    Returns:
        (B, 2 * C) -- every cell's ally count, then every cell's enemy count.
        Each living ship contributes exactly one unit, less whatever part of it
        falls on lattice sites this grid does not have.
    """

    width, height = world_size
    delta_x = ship_x - centre_x
    delta_y = ship_y - centre_y
    # Minimum image on the torus, as ``d - w * round(d / w)`` rather than the
    # usual ``(d + w/2) % w - w/2``. The two agree mathematically, but the
    # second adds half a 65536 px world to a delta of a few thousand, and
    # float32 near 32768 is only accurate to 0.004 -- enough error, once
    # divided by the playable radius and multiplied back up by the ring count,
    # to move a ship a sixty-fourth of a cell off its own centre. This form
    # leaves a delta smaller than the world untouched.
    delta_x = delta_x - width * torch.round(delta_x / width)
    delta_y = delta_y - height * torch.round(delta_y / height)
    # The grid is stored in units of the playable radius. A degenerate radius
    # (the unbounded combat arena) collapses every cell onto the origin, and the
    # target is not meaningful there; guard the division rather than emit NaN.
    unit = playable.clamp(min=1e-6)
    column, weight = barycentric_cells(delta_x / unit, delta_y / unit, rings, lookup)

    cells = 1 + 3 * rings * (rings + 1)
    weight = weight * contributes.unsqueeze(-1)  # (B, N, 3)
    # Column 0 of the scatter is a bin for everything off the grid, dropped
    # after. Ally and enemy are separate halves of one scatter so the ship axis
    # is reduced once.
    half = torch.where(is_ally.unsqueeze(-1), 0, cells + 1)
    target = (column + 1) + half  # (B, N, 3); 0 and cells+1 are the trash bins
    target = torch.where(column >= 0, target, half)
    scattered = torch.zeros(
        ship_x.shape[0], 2 * (cells + 1), device=ship_x.device, dtype=weight.dtype
    )
    scattered.scatter_add_(1, target.flatten(1), weight.flatten(1))
    return torch.cat((scattered[:, 1 : cells + 1], scattered[:, cells + 2 :]), dim=-1)


def hex_counts_from_state(
    state: TensorState,
    observer_team: torch.Tensor,
    rings: int,
    lookup: torch.Tensor,
    world_size: tuple[float, float],
) -> torch.Tensor:
    """The privileged count target for one observer per environment.

    Every living ship contributes, visible or not. The grid is centred on
    ``state.map_center`` and scaled by ``state.playable_boundary_radius``, both
    zero in the unbounded combat arena, where the target is not meaningful.

    Args:
        state:         Environment truth.
        observer_team: (B,) team whose allies fill the first half.
        rings:         Rings around the centre cell.
        lookup:        From :func:`hex_axial_lookup`, on ``state.device``.
        world_size:    (width, height) of the toroid.

    Returns:
        (B, 2 * C) -- see :func:`hex_counts`.
    """

    return hex_counts(
        state.ship_pos.real,
        state.ship_pos.imag,
        state.ship_team_id == observer_team.unsqueeze(1),
        state.ship_alive,
        state.map_center.real.unsqueeze(1),
        state.map_center.imag.unsqueeze(1),
        state.playable_boundary_radius.unsqueeze(1),
        world_size,
        rings,
        lookup,
    )


class HexDensityTarget:
    """The count target with its lookup resident on device, fused when asked.

    Args:
        world_size:   (width, height) of the toroid.
        device:       Where the lookup and every target live.
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
        self.rings = rings
        self.unit_cells = hex_cell_offsets(rings).to(device)  # (C, 2)
        self.lookup = hex_axial_lookup(rings).to(device)
        self.num_cells = self.unit_cells.shape[0]
        self._build = (
            hex_counts_from_state
            if compile_mode is None
            else torch.compile(hex_counts_from_state, mode=compile_mode, dynamic=False)
        )

    def __call__(self, state: TensorState, observer_team: torch.Tensor) -> torch.Tensor:
        """(B, 2 * C) count target for ``observer_team`` in each environment."""

        return self._build(state, observer_team, self.rings, self.lookup, self.world_size)
