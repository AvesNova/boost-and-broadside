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

from .modes.rules import ELIMINATION, ModeRules
from .physics.numerics import smootherstep
from .rules import MAP_SCALE_RADIUS, Rules


@contract
class Environment:
    """The rules that vary by environment, (B,) each, one value per pool: arena_radius (m)
    and map_scale (arena_radius / MAP_SCALE_RADIUS) float32; episode_ticks and respawn_ticks
    int32; the boundary's shrink (`boundary_fraction`): shrinking float32, 1 where the
    boundary shrinks (elimination) and 0 elsewhere, shrink_hold_ticks float32 (its hold, in
    ticks; 0 where it does not shrink) and shrink_law int32 (`ModeRules.shrink_law`)."""

    arena_radius: torch.Tensor
    map_scale: torch.Tensor
    episode_ticks: torch.Tensor
    respawn_ticks: torch.Tensor
    shrinking: torch.Tensor
    shrink_hold_ticks: torch.Tensor
    shrink_law: torch.Tensor


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
        """value(pool) for every environment of the pool (a number, or a 0-d tensor when
        the rules are live: `bnb.game.tuning.with_live`)."""
        parts = [
            torch.zeros((size,), dtype=dtype, device=device) + value(pool)
            for pool, size in zip(rules.pools, sizes)
        ]
        return parts[0] if len(parts) == 1 else torch.cat(parts)

    return Environment(
        arena_radius=per_pool(lambda pool: pool.arena_radius),
        map_scale=per_pool(lambda pool: pool.arena_radius / MAP_SCALE_RADIUS),
        episode_ticks=per_pool(lambda pool: pool.episode_ticks, torch.int32),
        respawn_ticks=per_pool(lambda pool: pool.respawn_ticks, torch.int32),
        shrinking=per_pool(lambda pool: float(shrinks(pool.mode))),
        shrink_hold_ticks=per_pool(
            lambda pool: (
                pool.mode.shrink_hold_seconds / rules.tick_length if shrinks(pool.mode) else 0.0
            )
        ),
        shrink_law=per_pool(
            lambda pool: pool.mode.shrink_law if shrinks(pool.mode) else 0, torch.int32
        ),
    )


def shrinks(mode: ModeRules) -> bool:
    """Whether the mode's boundary shrinks over the clock (elimination's)."""
    return mode.name == ELIMINATION


def boundary_fraction(arena: Environment, elapsed: torch.Tensor) -> torch.Tensor:
    """(B,) the boundary's radius over the arena radius after `elapsed` (B,) ticks of the
    episode. Where it shrinks (elimination) it holds at 1 for the hold, then falls to 0 at
    the clock's end: s = (elapsed - hold) / (episode - hold) in [0, 1], the fraction 1 - s
    (shrink law 0) or 1 - smootherstep(s) (law 1). Elsewhere 1. Every law is computed and
    the live law selects one, so it is data in the compiled tick."""
    total = arena.episode_ticks.to(torch.float32)
    hold = torch.minimum(arena.shrink_hold_ticks, total - 1.0).clamp_min(0.0)
    s = ((elapsed.to(torch.float32) - hold) / (total - hold).clamp_min(1.0)).clamp(0.0, 1.0)
    law = torch.where(arena.shrink_law == 1, smootherstep(s), s)
    return 1.0 - arena.shrinking * law


def boundary_time_to_zero(arena: Environment, elapsed: torch.Tensor, tick_length: float):
    """(B,) seconds until the boundary reaches radius 0 (the clock's end) where it shrinks,
    after `elapsed` (B,) ticks; 0 where it does not shrink."""
    left = (arena.episode_ticks - elapsed).clamp_min(0).to(torch.float32) * tick_length
    return arena.shrinking * left


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
