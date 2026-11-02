"""The batch's games (mixed spec 2): the per-environment rules as tensors, and the static
slices where each mode's own steps run.

Everything the games share (flight, bullets, hits, damage, sight, the observation) runs once
over the whole batch, reading the per-environment rules (`Environment`). Everything a mode
owns (its map, its phase, its spawns, its protection, its observation) runs on its own
slices only (`per_mode`), so no environment pays for another mode's rules, and the results
are written back into the batch tensors. The bounds are static: one graph.
"""

import dataclasses

import torch
from torch.utils import _pytree

from bnb.contracts.tree import contract

from .modes.rules import ModeRules
from .rules import MAP_SCALE_RADIUS, Rules


@contract
class Environment:
    """The rules that vary by environment, (B,) each, one value per pool: arena_radius (m)
    and map_scale (arena_radius / 2600) float32; episode_ticks and respawn_ticks int32."""

    arena_radius: torch.Tensor
    map_scale: torch.Tensor
    episode_ticks: torch.Tensor
    respawn_ticks: torch.Tensor


@dataclasses.dataclass(frozen=True)
class Slice:
    """Environments start:stop all play `mode`."""

    start: int
    stop: int
    mode: ModeRules


def sizes_in_batch(rules: Rules, batch: int) -> tuple[int, ...]:
    """The environments of each of the rules' pools in a batch of this size (the whole batch
    for one game without a size)."""
    if rules.pools[0].environments is None:
        return (batch,)
    sizes = tuple(pool.environments for pool in rules.pools)
    if sum(sizes) != batch:
        raise ValueError(f"the rules' pools hold {sum(sizes)} environments, not {batch}")
    return sizes


def slices(rules: Rules, batch: int) -> tuple[Slice, ...]:
    """The batch cut where the mode changes (adjacent pools of one mode share a slice)."""
    out: list[Slice] = []
    start = 0
    for pool, size in zip(rules.pools, sizes_in_batch(rules, batch)):
        if out and out[-1].mode == pool.mode:
            out[-1] = dataclasses.replace(out[-1], stop=start + size)
        else:
            out.append(Slice(start, start + size, pool.mode))
        start += size
    return tuple(out)


def environment(rules: Rules, batch: int, device=None) -> Environment:
    """The per-environment rules of a batch of this size (constants of the compiled graph)."""
    sizes = sizes_in_batch(rules, batch)

    def per_pool(value, dtype=torch.float32) -> torch.Tensor:
        """value(pool) for every environment of the pool."""
        parts = [
            torch.full((size,), value(pool), dtype=dtype, device=device)
            for pool, size in zip(rules.pools, sizes)
        ]
        return parts[0] if len(parts) == 1 else torch.cat(parts)

    return Environment(
        arena_radius=per_pool(lambda pool: pool.arena_radius),
        map_scale=per_pool(lambda pool: pool.arena_radius / MAP_SCALE_RADIUS),
        episode_ticks=per_pool(lambda pool: pool.episode_ticks, torch.int32),
        respawn_ticks=per_pool(lambda pool: pool.respawn_ticks, torch.int32),
    )


def per_mode(rules: Rules, batch: int, function, *trees):
    """function(mode, *trees) on each mode's slice of the batch, the results (pytrees of
    tensors with the batch leading) put back together. One slice is the whole batch, uncut."""
    parts = slices(rules, batch)
    if len(parts) == 1:
        return function(parts[0].mode, *trees)
    outputs = []
    for part in parts:
        cut = _pytree.tree_map_only(torch.Tensor, lambda x: x[part.start : part.stop], trees)
        outputs.append(function(part.mode, *cut))
    return _pytree.tree_map(lambda *pieces: torch.cat(pieces), *outputs)
