"""Visibility (game spec 10.1): which ships each team sees. Bullets are not perceived."""

import torch

from ..rules import Rules
from ..state import Fields, State, by_team, enemy


def core_radius(fields: Fields) -> torch.Tensor:
    """(B, M) each field's core radius (m): the inner edge of its band, the radius less half
    the band, and 0 for a field with no flat interior (game spec 10.1)."""
    return (fields.radius - 0.5 * fields.band_width).clamp_min(0.0)


def outer_radius(fields: Fields) -> torch.Tensor:
    """(B, M) each field's outer radius (m): the outer edge of its band, the radius plus
    half the band (game spec 10.1)."""
    return fields.radius + 0.5 * fields.band_width


def membership(discs: tuple, x: torch.Tensor, y: torch.Tensor) -> tuple:
    """Points as two (B, 2, P) int64 bitmasks: bit i of the first is set when the point is
    strictly inside core i, of the second when it is strictly inside field i's outer disc.
    discs: centre x, y, core radius^2 and outer radius^2, (B, M) each. With 64 fields bit
    63 is the sign bit; XOR, AND, NOT, the shifts below and this sum of disjoint bits all
    still read it correctly."""
    centre_x, centre_y, core_squared, outer_squared = discs
    M = centre_x.shape[1]
    offset_x = x[..., None] - centre_x[:, None, None, :]
    offset_y = y[..., None] - centre_y[:, None, None, :]
    distance_squared = offset_x * offset_x + offset_y * offset_y
    bits = torch.arange(M, device=x.device)
    in_core = ((distance_squared < core_squared[:, None, None, :]).long() << bits).sum(-1)
    in_field = ((distance_squared < outer_squared[:, None, None, :]).long() << bits).sum(-1)
    return in_core, in_field


def sighted(discs: tuple, observers: tuple, targets: tuple) -> torch.Tensor:
    """(B, 2, Q): some living observer of team T is within its range of target q with a
    clear sight line (game spec 10.1).

    discs: centre x, y, core radius^2 and outer radius^2, (B, M) each; observers: (x, y,
    in_core, in_field, alive, range), (B, 2, O); targets: (x, y, in_core, in_field),
    (B, 2, Q); in_core and in_field from `membership`.

    Each field splits the plane into its core, its band (core edge to outer edge) and
    outside. Field i blocks the line when
      - exactly one end is inside its core and the other is outside its outer disc:
        (core_a XOR core_b) AND NOT (field_a AND field_b) has bit i set (core to band and
        core to core see each other), or
      - neither end is inside its core and the segment passes within the core radius. A
        segment comes within rho of the centre exactly when the foot of the perpendicular
        from the centre falls inside it (0 < u.d < |d|^2, with u = centre - observer,
        d = target - observer) at a distance below rho (|u x d|^2 < rho^2 |d|^2), or an
        end is inside, which this case excludes.
    The fields are folded one at a time, so each pair stays in registers; M <= 64 keeps the
    unrolled loop small (compile and run time against M: see game-notes.md).
    """
    centre_x, centre_y, core_squared, _ = discs
    observer_x, observer_y, observer_core, observer_field, alive, vision = observers
    target_x, target_y, target_core, target_field = targets
    line_x = target_x[..., :, None] - observer_x[..., None, :]  # (B, 2, Q, O): d
    line_y = target_y[..., :, None] - observer_y[..., None, :]
    length_squared = line_x * line_x + line_y * line_y
    core_a, core_b = target_core[..., :, None], observer_core[..., None, :]
    field_a, field_b = target_field[..., :, None], observer_field[..., None, :]
    blocked = ((core_a ^ core_b) & ~(field_a & field_b)) != 0
    neither = ~(core_a | core_b)  # (B, 2, Q, O): bit i set when no end is in core i
    for i in range(centre_x.shape[1]):
        to_centre_x = centre_x[:, i, None, None, None] - observer_x[..., None, :]  # u
        to_centre_y = centre_y[:, i, None, None, None] - observer_y[..., None, :]
        along = to_centre_x * line_x + to_centre_y * line_y
        across = to_centre_x * line_y - to_centre_y * line_x
        blocked = blocked | (
            ((neither >> i) & 1 == 1)
            & (across * across < core_squared[:, i, None, None, None] * length_squared)
            & (along > 0.0)
            & (along < length_squared)
        )
    in_range = length_squared <= (vision * vision)[..., None, :]
    return (in_range & alive[..., None, :] & ~blocked).any(-1)


def vision_range(rules: Rules, state: State) -> torch.Tensor:
    """(B, N) each ship's vision range (m) as an observer: its type's, which reset and every
    resolve keep in `Match.vision_range`. Takes the rules like the api's other display
    functions."""
    return state.match.vision_range


def visibility(state: State) -> torch.Tensor:
    """(B, 2, N): ship n visible to team T, in sim order (game spec 10.1). Range tests use
    d^2 <= V^2 with the observer's V (`Match.vision_range`). In an environment without fog
    every living ship is visible."""
    fields, ships = state.fields, state.ships
    core, outer = core_radius(fields), outer_radius(fields)
    discs = (fields.x, fields.y, core * core, outer * outer)
    observer_x, observer_y = by_team(ships.x), by_team(ships.y)
    target_x, target_y = enemy(ships.x), enemy(ships.y)
    observers = (
        observer_x,
        observer_y,
        *membership(discs, observer_x, observer_y),
        by_team(ships.alive),
        by_team(state.match.vision_range),
    )
    seen = sighted(discs, observers, (target_x, target_y, *membership(discs, target_x, target_y)))
    unfogged = ~state.match.fogged[:, None, None]
    enemies = enemy(ships.alive) & (enemy(ships.fired) | enemy(ships.spawned) | seen | unfogged)
    own = torch.ones_like(enemies)  # (B, 2, S) each; a team always sees its own ships
    return torch.stack(
        [torch.cat([own[:, 0], enemies[:, 0]], -1), torch.cat([enemies[:, 1], own[:, 1]], -1)], 1
    )
