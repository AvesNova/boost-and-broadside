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


def pool_sizes(rules: Rules, batch: int) -> tuple[int, ...]:
    """Environments per pool of the rules, for a batch of this size."""
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
    for pool, size in zip(rules.pools, pool_sizes(rules, batch)):
        if out and out[-1].mode == pool.mode:
            out[-1] = dataclasses.replace(out[-1], stop=start + size)
        else:
            out.append(Slice(start, start + size, pool.mode))
        start += size
    return tuple(out)


def _per_pool(rules: Rules, batch: int, values: tuple, dtype, device) -> torch.Tensor:
    parts = [torch.full((size,), value, dtype=dtype, device=device)
             for value, size in zip(values, pool_sizes(rules, batch))]
    return parts[0] if len(parts) == 1 else torch.cat(parts)


def environment(rules: Rules, batch: int, device=None) -> Environment:
    """The per-environment rules of a batch of this size (constants of the compiled graph)."""
    pools = rules.pools

    def per_pool(name: str, dtype=torch.float32):
        return _per_pool(rules, batch, tuple(getattr(pool, name) for pool in pools), dtype,
                         device)

    return Environment(
        arena_radius=per_pool("arena_radius"),
        map_scale=_per_pool(rules, batch,
                            tuple(pool.arena_radius / MAP_SCALE_RADIUS for pool in pools),
                            torch.float32, device),
        episode_ticks=per_pool("episode_ticks", torch.int32),
        respawn_ticks=per_pool("respawn_ticks", torch.int32))


def per_mode(rules: Rules, batch: int, function, *trees):
    """function(mode, *trees) on each mode's slice of the batch, the results (pytrees of
    tensors with the batch leading) put back together. One slice is the whole batch, uncut."""
    parts = slices(rules, batch)
    if len(parts) == 1:
        return function(parts[0].mode, *trees)
    outputs = []
    for part in parts:
        cut = _pytree.tree_map_only(torch.Tensor, lambda x: x[part.start:part.stop], trees)
        outputs.append(function(part.mode, *cut))
    return _pytree.tree_map(lambda *pieces: torch.cat(pieces), *outputs)
