"""The index texture: ln n on a grid over the map, from the game's own medium
(`bnb.game.api.log_index`).

Fields are static within an episode, so the grid is built once per map. Only tiles that
touch a field's outer edge (radius plus half the band) are evaluated: beyond it a field
adds nothing (its profile and its gradient are exactly zero), so everywhere else ln n is
exactly 0. Each tile passes `medium` only the fields that touch it, padded with fields
that touch nothing, which leaves the sums and the product unchanged; a test holds the
grid to `medium` at sampled points.

A new map can be built a few tiles at a time (`IndexBuilder.run` with a time budget), so
a map change never stalls a frame for the whole build.
"""

import dataclasses
import math
import time

import numpy as np
import torch

from bnb.game import api

from .snapshot import Fields as FieldSnapshot

TEXEL_METRES = 2.0  # at map scale 1 (arena radius 2600 m); the texel grows with the map
TILE = 32  # texels per tile side
TILES_PER_CALL = 4  # about 1 to 3 ms per call at 5v5 and 50v50 on one thread


@dataclasses.dataclass(frozen=True)
class IndexGrid:
    """The square grid: texel (i, j) holds ln n at x = origin + (i + 0.5) texel, y = origin
    + (j + 0.5) texel. size texels per side, a whole number of tiles."""

    origin: float
    texel: float
    size: int


@dataclasses.dataclass(frozen=True)
class TileUpdate:
    """Values (TILE, TILE) float32 for the tile whose first texel is (column, row)."""

    column: int
    row: int
    values: np.ndarray


def grid_for(fields: FieldSnapshot, arena_radius: float) -> IndexGrid:
    """The grid covering the arena and every field, at TEXEL_METRES times the map scale."""
    texel = TEXEL_METRES * arena_radius / api.MAP_SCALE_RADIUS
    outer = fields.radius + 0.5 * fields.band_width
    reach = max(
        arena_radius,
        float(np.max(np.maximum(np.abs(fields.x), np.abs(fields.y)) + outer, initial=0.0)),
    )
    tiles = math.ceil(2.0 * reach / (texel * TILE))
    size = tiles * TILE
    return IndexGrid(origin=-0.5 * size * texel, texel=texel, size=size)


class IndexBuilder:
    """Builds one map's grid in tile batches. `run(budget)` evaluates batches until the
    budget (seconds) is spent and returns the finished tiles."""

    def __init__(self, fields: FieldSnapshot, arena_radius: float) -> None:
        self.grid = grid_for(fields, arena_radius)
        grid, tiles = self.grid, self.grid.size // TILE
        span = TILE * grid.texel
        low = grid.origin + np.arange(tiles) * span  # tile edges along either axis
        outer = fields.radius + 0.5 * fields.band_width

        # (M, tiles) the gap between each field's centre and each tile's edges, per axis
        def gap(centre: np.ndarray) -> np.ndarray:
            return np.clip(
                np.maximum(low[None] - centre[:, None], centre[:, None] - low[None] - span),
                0.0,
                None,
            )

        gap_x, gap_y = gap(fields.x), gap(fields.y)
        touches = gap_y[:, :, None] ** 2 + gap_x[:, None, :] ** 2 < outer[:, None, None] ** 2
        touches = touches.reshape(len(outer), -1).T  # (tiles^2, M), tile = row * tiles + column
        self.tiles = np.flatnonzero(touches.any(-1))
        self.touches = touches[self.tiles]
        self.fields = fields
        self.columns = tiles
        self.next = 0

    @property
    def done(self) -> bool:
        return self.next >= len(self.tiles)

    def _batch(self, start: int, stop: int) -> list[TileUpdate]:
        tiles, touches = self.tiles[start:stop], self.touches[start:stop]
        width = max(int(touches.sum(-1).max()), 1)
        # Each tile's touching fields first; the padding is a field far inside out.
        order = np.argsort(~touches, axis=-1, kind="stable")[:, :width]
        used = np.take_along_axis(touches, order, -1)

        def column(values: np.ndarray, padding: float, dtype) -> torch.Tensor:
            return torch.from_numpy(np.where(used, values[order], padding).astype(dtype))

        fields = self.fields
        tile_fields = (
            column(fields.x, 0.0, np.float32),
            column(fields.y, 0.0, np.float32),
            column(fields.radius, -1e9, np.float32),
            column(fields.band_width, 1.0, np.float32),
            column(fields.level, 1, np.int32),
        )
        grid = self.grid
        offsets = (np.arange(TILE) + 0.5) * grid.texel
        tile_x = grid.origin + (tiles % self.columns) * TILE * grid.texel
        tile_y = grid.origin + (tiles // self.columns) * TILE * grid.texel
        shape = (len(tiles), TILE, TILE)
        x = np.broadcast_to(tile_x[:, None, None] + offsets[None, None, :], shape)
        y = np.broadcast_to(tile_y[:, None, None] + offsets[None, :, None], shape)

        def points(values: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(values.reshape(len(tiles), -1).astype(np.float32))

        log_index = api.log_index(*tile_fields, points(x), points(y))
        values = log_index.numpy().reshape(len(tiles), TILE, TILE)
        return [
            TileUpdate(
                column=int(tile % self.columns) * TILE,
                row=int(tile // self.columns) * TILE,
                values=values[index],
            )
            for index, tile in enumerate(tiles)
        ]

    @torch.no_grad()
    def run(self, budget: float = math.inf) -> list[TileUpdate]:
        """Evaluate tile batches until done or `budget` seconds have passed (at least one
        batch per call)."""
        started, out = time.perf_counter(), []
        while not self.done:
            stop = min(self.next + TILES_PER_CALL, len(self.tiles))
            out.extend(self._batch(self.next, stop))
            self.next = stop
            if time.perf_counter() - started >= budget:
                break
        return out


def build(fields: FieldSnapshot, arena_radius: float) -> tuple[IndexGrid, np.ndarray]:
    """The whole grid at once: (grid, values (size, size) float32, row = y)."""
    builder = IndexBuilder(fields, arena_radius)
    values = np.zeros((builder.grid.size, builder.grid.size), dtype=np.float32)
    for update in builder.run():
        values[update.row : update.row + TILE, update.column : update.column + TILE] = update.values
    return builder.grid, values
