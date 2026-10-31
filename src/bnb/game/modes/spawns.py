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

from ..physics.spawning import Placement, formation, formation_rank
from ..rules import Rules
from ..state import ModeState, by_team
from .frontline import TEAM0_DEFENSE, TEAM0_SPAWN, TEAM1_DEFENSE, TEAM1_SPAWN
from .rules import FRONTLINE, TEAM_DEATHMATCH

INFINITY = float("inf")


def _nearest(x: torch.Tensor, y: torch.Tensor, target_x: torch.Tensor, target_y: torch.Tensor,
             counts: torch.Tensor) -> torch.Tensor:
    """Squared distance (..., K) from each candidate (..., K) to the nearest target (..., T)
    that counts (counts (..., T)); inf if none counts."""
    dx = x[..., None] - target_x[..., None, :]
    dy = y[..., None] - target_y[..., None, :]
    return torch.where(counts[..., None, :], dx * dx + dy * dy, INFINITY).amin(-1)


def _best(x: torch.Tensor, y: torch.Tensor, score: torch.Tensor):
    """The candidate (..., K) with the largest score, ties to the lower one: (x, y) (...)."""
    pick = score.argmax(-1, keepdim=True)
    return x.gather(-1, pick)[..., 0], y.gather(-1, pick)[..., 0]


def _farthest_from_enemies(x: torch.Tensor, y: torch.Tensor, target_x: torch.Tensor,
                           target_y: torch.Tensor, counts: torch.Tensor | None = None):
    """Each ship's candidate (B, N, K) farthest from its nearest enemy target (B, N; an
    infinite coordinate does not count). Only the other team's slots are compared
    (team-major: a flip of the team axis); counts (B, 2, S, S) [team, slot, enemy slot]
    restricts the pairs further."""
    B, N, K = x.shape
    # One stacked tensor, so the compiled stage stores the targets once instead of
    # recomputing them (a sqrt, a cos and a sin at a reset) for every pair: half the time.
    targets = by_team(torch.stack([target_x, target_y], -1)).flip(1)[:, :, None, None]
    dx = by_team(x)[..., None] - targets[..., 0]  # (B, 2, S, K, S)
    dy = by_team(y)[..., None] - targets[..., 1]
    distance = dx * dx + dy * dy
    if counts is not None:
        distance = torch.where(counts[:, :, :, None, :], distance, INFINITY)
    return _best(x, y, distance.amin(-1).reshape(B, N, K))


def _disc(radius: float, radius_uniform: torch.Tensor, angle_uniform: torch.Tensor):
    """Uniform points in a disc of this radius around the origin."""
    distance = radius * torch.sqrt(radius_uniform)
    angle = 2.0 * math.pi * angle_uniform
    return distance * torch.cos(angle), distance * torch.sin(angle)


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


def _group_centres(rules: Rules, uniforms: torch.Tensor):
    """(B, G) centres: centre 0 uniform in the disc of radius 0.85 R_a; each later one the
    best of K uniform draws there by distance to the nearest earlier centre."""
    mode = rules.mode
    B, G, K = uniforms.shape[0], len(mode.spawn_groups), mode.spawn_candidates
    drawn = uniforms.reshape(B, G, K, 2)
    x, y = _disc(mode.group_centre_fraction * rules.arena_radius, drawn[..., 0], drawn[..., 1])
    centres_x, centres_y = [x[:, 0, 0]], [y[:, 0, 0]]
    for g in range(1, G):
        earlier_x, earlier_y = torch.stack(centres_x, -1), torch.stack(centres_y, -1)
        counts = torch.ones((1, g), dtype=torch.bool, device=uniforms.device)
        nearest = _nearest(x[:, g], y[:, g], earlier_x, earlier_y, counts)  # (B, K)
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
    return (torch.where(chosen, x[:, None, :], 0.0).sum(-1),
            torch.where(chosen, y[:, None, :], 0.0).sum(-1), count > 0)


def deathmatch(rules: Rules, uniforms: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
               alive: torch.Tensor, reset: torch.Tensor, keys: torch.Tensor) -> Placement:
    """Team deathmatch spawns, uniform or grouped (modes spec 4).

    uniforms: the mode's tick uniforms (B, U_mode); x, y, alive (B, N): ship positions and
    who is alive after the damage phase; reset (B,): environments starting a match, whose
    ships all spawn together; keys (B, N): the formation keys (grouped membership).
    """
    mode = rules.mode
    B, N, K = x.shape[0], rules.num_ships, mode.spawn_candidates
    groups = mode.spawn_groups
    per_environment = 2 * K * len(groups)
    per_ship = uniforms[:, per_environment:].reshape(B, N, -1)
    radius_uniform, angle_uniform = per_ship[..., :K], per_ship[..., K:2 * K]
    starting = reset[:, None]

    # Uniform over the arena: the respawn rule, and the grouped rule's fallback.
    uniform_x, uniform_y = _disc(rules.arena_radius, radius_uniform, angle_uniform)
    if groups:
        centres_x, centres_y = _group_centres(rules, uniforms[:, :per_environment])
        group = _group_of_rank(groups, formation_rank(rules, keys))  # (B, N)
        offset_x, offset_y = _disc(mode.group_radius, radius_uniform, angle_uniform)
        reset_x = centres_x.gather(1, group)[..., None] + offset_x
        reset_y = centres_y.gather(1, group)[..., None] + offset_y
        by_group = by_team(group)
        rival = by_group[..., :, None] == by_group.flip(1)[..., None, :]  # same-group enemies
        counts = rival | ~reset[:, None, None, None]
        anchor_x, anchor_y, any_living = _anchors(alive, per_ship[..., 2 * K + 1], x, y)
        distance = mode.anchor_distance_min \
            + (mode.anchor_distance_max - mode.anchor_distance_min) * radius_uniform
        angle = 2.0 * math.pi * angle_uniform
        respawn_x = torch.where(any_living[..., None], anchor_x[..., None]
                                + distance * torch.cos(angle), uniform_x)
        respawn_y = torch.where(any_living[..., None], anchor_y[..., None]
                                + distance * torch.sin(angle), uniform_y)
        candidate_x = torch.where(starting[..., None], reset_x, respawn_x)
        candidate_y = torch.where(starting[..., None], reset_y, respawn_y)
    else:
        counts, candidate_x, candidate_y = None, uniform_x, uniform_y

    # One pass serves both rules. A reset is symmetric: every ship's candidate 0 is its
    # provisional point, and it keeps away from the provisional points of its rivals (the
    # enemies, of its own group when grouped). A respawn keeps away from the living enemies.
    target = starting | alive
    target_x = torch.where(target, torch.where(starting, candidate_x[..., 0], x), INFINITY)
    target_y = torch.where(target, torch.where(starting, candidate_y[..., 0], y), INFINITY)
    place_x, place_y = _farthest_from_enemies(candidate_x, candidate_y, target_x, target_y,
                                              counts)
    heading_cos, heading_sin = _heading(per_ship[..., 2 * K])
    return Placement(x=place_x, y=place_y, heading_cos=heading_cos, heading_sin=heading_sin)


def frontline(rules: Rules, uniforms: torch.Tensor, mode: ModeState) -> Placement:
    """Frontline spawns: a uniform point in the disc of 0.35 zone radii around the team's
    current spawn zone, nose toward the zone it attacks (the enemy's defense zone).
    mode: the state after this tick's captures (or a new match's)."""
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
    radius = rules.mode.spawn_zone_fraction * rules.mode.zone_radius * rules.map_scale
    offset_x, offset_y = _disc(radius, per_ship[..., 0], per_ship[..., 1])
    x, y = spawn_x + offset_x, spawn_y + offset_y
    to_x, to_y = target_x - x, target_y - y
    length = torch.sqrt(to_x * to_x + to_y * to_y).clamp_min(1e-6)
    return Placement(x=x, y=y, heading_cos=to_x / length, heading_sin=to_y / length)


def placement(rules: Rules, uniforms, mode: ModeState, x, y, alive, reset, axis_x, axis_y,
              rotation, keys) -> Placement:
    """The mode's placement for every ship (selected later by the spawn mask)."""
    if rules.mode.name == TEAM_DEATHMATCH:
        return deathmatch(rules, uniforms, x, y, alive, reset, keys)
    if rules.mode.name == FRONTLINE:
        return frontline(rules, uniforms, mode)
    return formation(rules, axis_x, axis_y, rotation, keys)  # sandbox: game spec 8's
