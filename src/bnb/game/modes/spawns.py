"""Mode-owned spawn placement (modes spec 4): where each spawning ship appears and which way
it points. The game's spawn state (speed, energy, shield, waits, the medium) is applied by
`physics.spawning.spawn`.

Every placement is computed for every ship, every tick, as one static-shape pass, and the
spawn mask selects it. Candidates are compared by squared distance to the nearest enemy
that counts; ties go to the lower candidate, and with nothing to count candidate 0 wins.
Nothing depends on the team index or the sim index beyond each ship's own team.
"""

import math

import torch

from ..physics.spawning import Placement, formation, formation_rank, uniform_disc
from ..rules import Rules
from ..state import ModeState, by_team, enemy
from .frontline import TEAM0_DEFENSE, TEAM0_SPAWN, TEAM1_DEFENSE, TEAM1_SPAWN
from .rules import FRONTLINE, TEAM_DEATHMATCH, ModeRules

INFINITY = float("inf")


def _nearest(
    x: torch.Tensor,
    y: torch.Tensor,
    target_x: torch.Tensor,
    target_y: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    """Squared distance (..., C) from each candidate (..., C) to the nearest target (...,
    targets) that counts (counts (..., targets)); inf if none counts."""
    dx = x[..., None] - target_x[..., None, :]
    dy = y[..., None] - target_y[..., None, :]
    return torch.where(counts[..., None, :], dx * dx + dy * dy, INFINITY).amin(-1)


def _best(x: torch.Tensor, y: torch.Tensor, score: torch.Tensor):
    """The candidate (..., C) with the largest score, ties to the lower one: (x, y) (...)."""
    pick = score.argmax(-1, keepdim=True)
    return x.gather(-1, pick)[..., 0], y.gather(-1, pick)[..., 0]


def _farthest_from_enemies(
    x: torch.Tensor,
    y: torch.Tensor,
    target_x: torch.Tensor,
    target_y: torch.Tensor,
    counts: torch.Tensor | None = None,
):
    """Each ship's candidate (B, N, C) farthest from its nearest enemy target (B, N; an
    infinite coordinate does not count). Only the other team's slots are compared
    (team-major: a flip of the team axis); counts (B, 2, S, S) [team, slot, enemy slot]
    restricts the pairs further."""
    B, N, candidates = x.shape
    # One stacked tensor, so the compiled stage stores the targets once instead of
    # recomputing them (a sqrt, a cos and a sin at a reset) for every pair: half the time.
    targets = enemy(torch.stack([target_x, target_y], -1))[:, :, None, None]
    dx = by_team(x)[..., None] - targets[..., 0]  # (B, 2, S, C, S)
    dy = by_team(y)[..., None] - targets[..., 1]
    distance = dx * dx + dy * dy
    if counts is not None:
        distance = torch.where(counts[:, :, :, None, :], distance, INFINITY)
    return _best(x, y, distance.amin(-1).reshape(B, N, candidates))


def _clip_to_disc(x: torch.Tensor, y: torch.Tensor, radius: torch.Tensor):
    """Points clipped radially into the disc of this radius (broadcasting against the
    points) around the origin."""
    distance = torch.sqrt(x * x + y * y)
    scale = torch.where(distance > radius, radius / distance.clamp_min(1e-6), 1.0)
    return x * scale, y * scale


def _heading(uniform: torch.Tensor):
    angle = 2.0 * math.pi * uniform
    return torch.cos(angle), torch.sin(angle)


def _group_of_rank(groups: tuple[int, ...], rank: torch.Tensor) -> torch.Tensor:
    """The group (by formation rank, spawn_groups[g] ships of each team in group g)."""
    group = torch.zeros_like(rank)
    boundary = 0
    for size in groups[:-1]:
        boundary += size
        group = group + (rank >= boundary).to(rank.dtype)
    return group


def _group_centres(mode_rules: ModeRules, arena_radius: torch.Tensor, uniforms: torch.Tensor):
    """(B, G) centres: centre 0 uniform in the disc of radius group_centre_fraction R_a
    (arena_radius (B,)); each later one the best of C uniform draws there by distance to the
    nearest earlier centre."""
    B, G = uniforms.shape[0], len(mode_rules.spawn_groups)
    drawn = uniforms.reshape(B, G, mode_rules.spawn_candidates, 2)
    x, y = uniform_disc(
        (mode_rules.group_centre_fraction * arena_radius)[:, None, None],
        drawn[..., 0],
        drawn[..., 1],
    )
    centres_x, centres_y = [x[:, 0, 0]], [y[:, 0, 0]]
    for g in range(1, G):
        earlier_x, earlier_y = torch.stack(centres_x, -1), torch.stack(centres_y, -1)
        counts = torch.ones((1, g), dtype=torch.bool, device=uniforms.device)
        nearest = _nearest(x[:, g], y[:, g], earlier_x, earlier_y, counts)  # (B, C)
        best_x, best_y = _best(x[:, g], y[:, g], nearest)
        centres_x.append(best_x)
        centres_y.append(best_y)
    return torch.stack(centres_x, -1), torch.stack(centres_y, -1)


def _anchors(alive: torch.Tensor, uniform: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
    """For every ship a uniformly random living ship (a spawning ship is not living):
    anchor (x, y) (B, N), and whether any ship lives (B, 1)."""
    living = alive.to(torch.int32)
    count = living.sum(-1, keepdim=True)  # (B, 1)
    choice = (uniform * count).to(torch.int32).clamp_max((count - 1).clamp_min(0))  # (B, N)
    order = living.cumsum(-1) - 1  # the rank of each living ship
    chosen = (order[:, None, :] == choice[..., None]) & alive[:, None, :]  # (B, N, N)
    return (
        torch.where(chosen, x[:, None, :], 0.0).sum(-1),
        torch.where(chosen, y[:, None, :], 0.0).sum(-1),
        count > 0,
    )


def deathmatch_placement(
    rules: Rules,
    mode_rules: ModeRules,
    arena_radius: torch.Tensor,
    uniforms: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    alive: torch.Tensor,
    reset: torch.Tensor,
    keys: torch.Tensor,
) -> Placement:
    """Team deathmatch spawns, uniform or grouped (modes spec 4).

    arena_radius (B,); uniforms: the mode's tick uniforms (B, U_mode); x, y, alive (B, N):
    ship positions and who is alive after the damage phase; reset (B,): environments
    starting a match, whose ships all spawn together; keys (B, N): the formation keys
    (grouped membership).
    """
    B, N, candidates = x.shape[0], rules.num_ships, mode_rules.spawn_candidates
    groups = mode_rules.spawn_groups
    per_environment = 2 * candidates * len(groups)
    per_ship = uniforms[:, per_environment:].reshape(B, N, -1)
    radius_uniform, angle_uniform = (
        per_ship[..., :candidates],
        per_ship[..., candidates : 2 * candidates],
    )
    starting = reset[:, None]

    # Uniform over the arena: the respawn rule, and the grouped rule's fallback.
    radius = arena_radius[:, None, None]
    uniform_x, uniform_y = uniform_disc(radius, radius_uniform, angle_uniform)
    if groups:
        centres_x, centres_y = _group_centres(
            mode_rules, arena_radius, uniforms[:, :per_environment]
        )
        group = _group_of_rank(groups, formation_rank(rules, keys))  # (B, N)
        offset_x, offset_y = uniform_disc(mode_rules.group_radius, radius_uniform, angle_uniform)
        reset_x = centres_x.gather(1, group)[..., None] + offset_x
        reset_y = centres_y.gather(1, group)[..., None] + offset_y
        rival = by_team(group)[..., :, None] == enemy(group)[..., None, :]  # same-group enemies
        counts = rival | ~reset[:, None, None, None]
        anchor_x, anchor_y, any_living = _anchors(alive, per_ship[..., 2 * candidates + 1], x, y)
        distance = (
            mode_rules.anchor_distance_min
            + (mode_rules.anchor_distance_max - mode_rules.anchor_distance_min) * radius_uniform
        )
        angle = 2.0 * math.pi * angle_uniform
        respawn_x = torch.where(
            any_living[..., None], anchor_x[..., None] + distance * torch.cos(angle), uniform_x
        )
        respawn_y = torch.where(
            any_living[..., None], anchor_y[..., None] + distance * torch.sin(angle), uniform_y
        )
        # A group near the edge, or an anchor there, can put a candidate outside the arena:
        # every grouped candidate is clipped into the disc R_a - spawn_margin. (The uniform rule's
        # candidates are inside the arena by construction.)
        candidate_x, candidate_y = _clip_to_disc(
            torch.where(starting[..., None], reset_x, respawn_x),
            torch.where(starting[..., None], reset_y, respawn_y),
            (radius - mode_rules.spawn_margin).clamp_min(0.0),
        )
    else:
        counts, candidate_x, candidate_y = None, uniform_x, uniform_y

    # One pass serves both rules. A reset is symmetric: every ship's candidate 0 is its
    # provisional point, and it keeps away from the provisional points of its rivals (the
    # enemies, of its own group when grouped). A respawn keeps away from the living enemies.
    target = starting | alive
    target_x = torch.where(target, torch.where(starting, candidate_x[..., 0], x), INFINITY)
    target_y = torch.where(target, torch.where(starting, candidate_y[..., 0], y), INFINITY)
    place_x, place_y = _farthest_from_enemies(candidate_x, candidate_y, target_x, target_y, counts)
    heading_cos, heading_sin = _heading(per_ship[..., 2 * candidates])
    return Placement(x=place_x, y=place_y, heading_cos=heading_cos, heading_sin=heading_sin)


def frontline_placement(
    rules: Rules,
    mode_rules: ModeRules,
    map_scale: torch.Tensor,
    uniforms: torch.Tensor,
    mode: ModeState,
) -> Placement:
    """Frontline spawns: a uniform point in the disc of spawn_zone_fraction zone radii around
    the team's current spawn zone, nose toward the zone it attacks (the enemy's defense zone).
    map_scale (B,); mode: the state after this tick's captures (or a new match's)."""
    B, N, S = uniforms.shape[0], rules.num_ships, rules.ships_per_team
    per_ship = uniforms[:, 2:].reshape(B, N, 2)

    def centre(roles: tuple[int, int]):
        """(B, N) x, y of the zone with each ship's team's role (roles by team)."""
        mine = torch.stack([mode.role == role for role in roles], 1)  # (B, 2, Z)
        zone_x = torch.where(mine, mode.zone_x[:, None], 0.0).sum(-1)
        zone_y = torch.where(mine, mode.zone_y[:, None], 0.0).sum(-1)
        return zone_x.repeat_interleave(S, 1), zone_y.repeat_interleave(S, 1)

    spawn_x, spawn_y = centre((TEAM0_SPAWN, TEAM1_SPAWN))
    target_x, target_y = centre((TEAM1_DEFENSE, TEAM0_DEFENSE))
    radius = (mode_rules.spawn_zone_fraction * mode_rules.zone_radius * map_scale)[:, None]
    offset_x, offset_y = uniform_disc(radius, per_ship[..., 0], per_ship[..., 1])
    x, y = spawn_x + offset_x, spawn_y + offset_y
    to_x, to_y = target_x - x, target_y - y
    length = torch.sqrt(to_x * to_x + to_y * to_y).clamp_min(1e-6)
    return Placement(x=x, y=y, heading_cos=to_x / length, heading_sin=to_y / length)


def placement(
    rules: Rules,
    mode_rules: ModeRules,
    arena_radius,
    map_scale,
    uniforms,
    mode: ModeState,
    x,
    y,
    alive,
    reset,
    axis_x,
    axis_y,
    rotation,
    keys,
) -> Placement:
    """The mode's placement for every ship (selected later by the spawn mask); arena_radius
    and map_scale (B,)."""
    if mode_rules.name == TEAM_DEATHMATCH:
        return deathmatch_placement(
            rules, mode_rules, arena_radius, uniforms, x, y, alive, reset, keys
        )
    if mode_rules.name == FRONTLINE:
        return frontline_placement(rules, mode_rules, map_scale, uniforms, mode)
    return formation(
        rules, arena_radius, map_scale, axis_x, axis_y, rotation, keys
    )  # sandbox: game spec 8's
