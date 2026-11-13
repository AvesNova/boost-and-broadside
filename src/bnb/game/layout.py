"""The batch's games (mixed spec 2): the per-environment rules as tensors, and the static
slices where each mode's own steps run.

Everything the games share (flight, bullets, hits, damage, sight, the observation) runs once
over the whole batch, reading the per-environment rules (`Environment`). Everything a mode
owns (its map, its phase, its spawns, its protection, its observation) runs on its own
slices only (`per_mode`), so no environment pays for another mode's rules, and the results
are written back into the batch tensors. The bounds are static: one graph.
"""

import dataclasses
import functools
import math

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
    ticks; 0 where it does not shrink), shrink_end_ticks float32 (the tick it reaches its
    end, `ModeRules.shrink_end_lead_seconds` before the clock's), shrink_end_fraction
    float32 (its end: minus the ramp width over the arena radius, so even the centre lies a
    ramp width beyond it), shrink_zero_ticks float32 (the tick it passes radius 0) and
    shrink_law int32 (`ModeRules.shrink_law`). Map-size sampling (`PoolRules.arena_scale`,
    `draw_arena_scale`): scale_sigma float32, the log-normal's sigma (0 where the pool plays
    its radius every episode), scale_cdf_low and scale_cdf_high float32, the standard
    normal CDF at its bounds' ln over sigma (0 and 1 where it does not sample), and
    minimum_radius float32 (m), the mode's minimum arena radius a sampled episode keeps.
    `environment` gives the pools' own radii; `sized` the episodes' (each environment's
    drawn scale, `Match.arena_scale`)."""

    arena_radius: torch.Tensor
    map_scale: torch.Tensor
    episode_ticks: torch.Tensor
    respawn_ticks: torch.Tensor
    shrinking: torch.Tensor
    shrink_hold_ticks: torch.Tensor
    shrink_end_ticks: torch.Tensor
    shrink_end_fraction: torch.Tensor
    shrink_zero_ticks: torch.Tensor
    shrink_law: torch.Tensor
    scale_sigma: torch.Tensor
    scale_cdf_low: torch.Tensor
    scale_cdf_high: torch.Tensor
    minimum_radius: torch.Tensor


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _scale_numbers(pool) -> tuple[float, float, float]:
    """(sigma, CDF at ln(low) / sigma, CDF at ln(high) / sigma) of a pool's map-size
    sampling; (0, 0, 1) without."""
    scale = pool.arena_scale
    if scale is None:
        return 0.0, 0.0, 1.0
    return (
        scale.sigma,
        _normal_cdf(math.log(scale.low) / scale.sigma),
        _normal_cdf(math.log(scale.high) / scale.sigma),
    )


# The draw's probability is kept this far inside (0, 1), so the inverse CDF stays finite.
_PROBABILITY_MARGIN = 1e-6


def draw_arena_scale(arena: Environment, uniform: torch.Tensor) -> torch.Tensor:
    """(B,) each environment's arena scale from one uniform in [0, 1) (B,): the log-normal
    ln(scale) ~ Normal(0, sigma) truncated to its bounds, by the inverse CDF (p between the
    CDF at the bounds, then scale = exp(sigma ndtri(p))): one draw, no rejection, no
    branch. 1 where the pool does not sample (sigma 0)."""
    low, high = arena.scale_cdf_low, arena.scale_cdf_high
    probability = (low + uniform * (high - low)).clamp(
        _PROBABILITY_MARGIN, 1.0 - _PROBABILITY_MARGIN
    )
    scale = torch.exp(arena.scale_sigma * torch.special.ndtri(probability))
    return torch.where(arena.scale_sigma > 0.0, scale, torch.ones_like(scale))


def sized(arena: Environment, arena_scale: torch.Tensor) -> Environment:
    """The environments' rules at their episodes' arena scale (B,): where the pool samples,
    the arena radius is its own times the scale, at least the mode's minimum (as
    `bnb.game.api.arena_radius_for` derives a radius), and the map scale follows; elsewhere
    the pool's own radius. Every map length that scales with the map (fields, zones, spawns,
    the boundary's ramp) reads this map scale."""
    sampled = torch.maximum(arena.arena_radius * arena_scale, arena.minimum_radius)
    radius = torch.where(arena.scale_sigma > 0.0, sampled, arena.arena_radius)
    return dataclasses.replace(arena, arena_radius=radius, map_scale=radius / MAP_SCALE_RADIUS)


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

    episode_ticks = per_pool(lambda pool: pool.episode_ticks, torch.int32)
    shrinking = per_pool(lambda pool: float(shrinks(pool.mode)))
    shrink_law = per_pool(
        lambda pool: pool.mode.shrink_law if shrinks(pool.mode) else 0, torch.int32
    )
    # The shrink's schedule: it ends `shrink_end_lead_seconds` before the clock (at least a
    # tick after the match's start), its hold at most a tick before its end.
    total = episode_ticks.to(torch.float32)
    lead = per_pool(lambda pool: pool.mode.shrink_end_lead_seconds / rules.tick_length)
    end = torch.maximum(total - lead, torch.ones_like(total))
    hold = per_pool(lambda pool: pool.mode.shrink_hold_seconds / rules.tick_length)
    hold = torch.minimum(hold, end - 1.0).clamp_min(0.0)
    # its end: a ramp width (m at map scale 1, times the map scale) inside the centre, as a
    # fraction of the arena radius (the map scale cancels: the ramp over MAP_SCALE_RADIUS)
    end_fraction = shrinking * (-rules.boundary_ramp_width / MAP_SCALE_RADIUS)
    # the zero crossing's share of the shrink (`zero_share`)
    s_zero = per_pool(lambda pool: zero_share(pool.mode.shrink_law, rules.boundary_ramp_width))
    return Environment(
        arena_radius=per_pool(lambda pool: pool.arena_radius),
        map_scale=per_pool(lambda pool: pool.arena_radius / MAP_SCALE_RADIUS),
        episode_ticks=episode_ticks,
        respawn_ticks=per_pool(lambda pool: pool.respawn_ticks, torch.int32),
        shrinking=shrinking,
        shrink_hold_ticks=shrinking * hold,
        shrink_end_ticks=torch.where(shrinking > 0, end, total),
        shrink_end_fraction=end_fraction,
        shrink_zero_ticks=torch.where(shrinking > 0, hold + s_zero * (end - hold), total),
        shrink_law=shrink_law,
        scale_sigma=per_pool(lambda pool: _scale_numbers(pool)[0]),
        scale_cdf_low=per_pool(lambda pool: _scale_numbers(pool)[1]),
        scale_cdf_high=per_pool(lambda pool: _scale_numbers(pool)[2]),
        minimum_radius=per_pool(lambda pool: pool.minimum_arena_radius),
    )


def zero_share(law, ramp_width):
    """The share s of the shrink at which the boundary passes radius 0: its law reaches
    1 / (1 + ramp_width / MAP_SCALE_RADIUS) there (s itself for law 0; the smootherstep's
    inverse for law 1, by a fixed bisection). Plain numbers (static rules) are solved once
    and kept; 0-d tensors (live rules) by the same bisection in tensors, branch-free."""
    if isinstance(law, torch.Tensor) or isinstance(ramp_width, torch.Tensor):
        reach = 1.0 / (1.0 + torch.as_tensor(ramp_width, dtype=torch.float32) / MAP_SCALE_RADIUS)
        low, high = torch.zeros_like(reach), torch.ones_like(reach)
        for _ in range(_BISECTIONS):
            middle = 0.5 * (low + high)
            below = smootherstep(middle) < reach
            low, high = torch.where(below, middle, low), torch.where(below, high, middle)
        return torch.where(torch.as_tensor(law) == 1, 0.5 * (low + high), reach)
    return _zero_share(int(law), float(ramp_width))


_BISECTIONS = 24  # to 2^-24 of the shrink, float32's resolution


@functools.lru_cache(maxsize=64)
def _zero_share(law: int, ramp_width: float) -> float:
    reach = 1.0 / (1.0 + ramp_width / MAP_SCALE_RADIUS)
    if law != 1:
        return reach
    low, high = 0.0, 1.0
    for _ in range(_BISECTIONS):
        middle = 0.5 * (low + high)
        if middle**3 * (middle * (6.0 * middle - 15.0) + 10.0) < reach:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def shrinks(mode: ModeRules) -> bool:
    """Whether the mode's boundary shrinks over the clock (elimination's)."""
    return mode.name == ELIMINATION


def boundary_fraction(arena: Environment, elapsed: torch.Tensor) -> torch.Tensor:
    """(B,) the boundary's radius over the arena radius after `elapsed` (B,) ticks of the
    episode. Where it shrinks (elimination) it holds at 1 for the hold, then falls past 0 to
    its end fraction e (minus a ramp width over the arena radius) at its end, the clock's
    end less the lead: s = (elapsed - hold) / (end - hold) in [0, 1], the fraction
    1 - (1 - e) s (shrink law 0) or 1 - (1 - e) smootherstep(s) (law 1), so from its end
    every ship, the centre's too, is a full ramp width beyond it. Negative once it passes 0
    (the damage reads it so; the observation shows it clamped at 0). Elsewhere 1. Every law
    is computed and the live law selects one, so it is data in the compiled tick."""
    hold, end = arena.shrink_hold_ticks, arena.shrink_end_ticks
    s = ((elapsed.to(torch.float32) - hold) / (end - hold).clamp_min(1.0)).clamp(0.0, 1.0)
    law = torch.where(arena.shrink_law == 1, smootherstep(s), s)
    return 1.0 - arena.shrinking * (1.0 - arena.shrink_end_fraction) * law


def boundary_time_to_zero(arena: Environment, elapsed: torch.Tensor, tick_length: float):
    """(B,) seconds until the boundary reaches radius 0 where it shrinks, after `elapsed`
    (B,) ticks (0 once it has); 0 where it does not shrink."""
    left = (arena.shrink_zero_ticks - elapsed.to(torch.float32)).clamp_min(0.0) * tick_length
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
