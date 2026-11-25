"""The adaption layer's index texture (filled into v2's IndexGrid by the caller): ln n on a square grid over the map, from the game's own medium
(lean.game.core.medium), for the fields' topographic fill and contours.

Fields are fixed within an episode, so the grid is built once per deal. Only tiles that touch
a field's outer disc (radius plus half the band) are evaluated: beyond it a field adds
nothing, so ln n is exactly 0 everywhere else. Building reads torch, so it runs where the game
runs (the video's process, the live window's game process), never in a frame.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import torch

from lean.game import core

from .snapshot import Fields

TEXEL_METRES = 2.0 * core.GameConfig().arena_radius / core.MAP_SCALE_RADIUS  # v2's 2 m at scale 1
TILE = 32  # texels per tile side
TILES_PER_BATCH = 64


@dataclasses.dataclass(frozen=True)
class IndexGrid:
    """The square grid: texel (i, j) holds ln n at x = origin + (i + 0.5) texel, y = origin +
    (j + 0.5) texel; size texels per side, a whole number of tiles. values (size, size)
    float32, row = y."""

    origin: float
    texel: float
    size: int
    values: np.ndarray


def _grid_size(fields: Fields, arena_radius: float, texel: float) -> int:
    reach = max(
        arena_radius,
        float(np.max(np.maximum(np.abs(fields.x), np.abs(fields.y)) + fields.outer_radius,
                     initial=0.0)),
    )
    return math.ceil(2.0 * reach / (texel * TILE)) * TILE


@torch.no_grad()
def build(fields: Fields, arena_radius: float, texel: float = TEXEL_METRES) -> IndexGrid:
    """The whole grid for these fields (an empty grid of zeros with no fields)."""
    size = _grid_size(fields, arena_radius, texel)
    origin = -0.5 * size * texel
    values = np.zeros((size, size), np.float32)
    if len(fields.x) == 0:
        return IndexGrid(origin, texel, size, values)
    tiles = size // TILE
    span = TILE * texel
    low = origin + np.arange(tiles) * span

    def gap(centre: np.ndarray) -> np.ndarray:  # (F, tiles): field centre to tile, per axis
        return np.clip(
            np.maximum(low[None] - centre[:, None], centre[:, None] - low[None] - span), 0.0, None
        )

    gap_x, gap_y = gap(fields.x), gap(fields.y)
    touches = gap_y[:, :, None] ** 2 + gap_x[:, None, :] ** 2 < fields.outer_radius[:, None, None] ** 2
    touched = np.flatnonzero(touches.any(0).reshape(-1))  # tile = row * tiles + column
    rows = torch.tensor(
        np.stack(
            [fields.x, fields.y, fields.radius, fields.band_width, fields.log_index]
        ),
        dtype=torch.float32,
    )[None]  # (1, 5, F) as State.fields
    offsets = (np.arange(TILE) + 0.5) * texel
    for start in range(0, len(touched), TILES_PER_BATCH):
        batch = touched[start : start + TILES_PER_BATCH]
        column, row = batch % tiles, batch // tiles
        x = origin + column[:, None, None] * span + offsets[None, None, :]
        y = origin + row[:, None, None] * span + offsets[None, :, None]
        x, y = np.broadcast_arrays(x, y)
        log_index, _, _ = core.medium(
            rows,
            torch.tensor(x.reshape(1, -1), dtype=torch.float32),
            torch.tensor(y.reshape(1, -1), dtype=torch.float32),
        )
        tile_values = log_index.numpy().reshape(len(batch), TILE, TILE)
        for k, (c, r) in enumerate(zip(column, row, strict=True)):
            values[r * TILE : (r + 1) * TILE, c * TILE : (c + 1) * TILE] = tile_values[k]
    return IndexGrid(origin, texel, size, values)
