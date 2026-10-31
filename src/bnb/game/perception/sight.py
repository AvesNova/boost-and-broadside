"""Visibility (game spec 10.1): which ships each team sees. Bullets are not perceived."""

import torch

from ..rules import Rules
from ..state import State, by_team


def enemy(values: torch.Tensor) -> torch.Tensor:
    """(B, N) per-ship values as seen by each team's enemy: row T holds team 1 - T."""
    return by_team(values).flip(1)


def core_membership(cores: tuple, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """(B, 2, P) int64 points as bitmasks: bit i is set when the point is strictly inside
    core i. With 64 cores bit 63 is the sign bit; XOR, the shifts below and this sum of
    disjoint bits all still read it correctly."""
    centre_x, centre_y, radius_squared = cores
    M = centre_x.shape[1]
    offset_x = x[..., None] - centre_x[:, None, None, :]
    offset_y = y[..., None] - centre_y[:, None, None, :]
    inside = offset_x * offset_x + offset_y * offset_y < radius_squared[:, None, None, :]
    return (inside.long() << torch.arange(M, device=x.device)).sum(-1)


def sighted(cores: tuple, observers: tuple, targets: tuple) -> torch.Tensor:
    """(B, 2, Q): some living observer of team T is within its range of target q with a
    clear sight line (game spec 10.1).

    cores: centre x, y and radius^2, (B, M); observers: (x, y, membership, alive, range),
    (B, 2, O); targets: (x, y, membership), (B, 2, Q); membership from `core_membership`.

    A segment comes within rho of a core centre exactly when an end is inside the core, or
    the foot of the perpendicular from the centre falls inside the segment (0 < u.d < |d|^2,
    with u = centre - observer, d = target - observer) at a distance below rho
    (|u x d|^2 < rho^2 |d|^2). With the exception for two ends inside the same core, a
    core blocks the line when
      - exactly one end is inside it: the bitmasks differ in its bit, or
      - neither end is inside and the perpendicular test holds. Giving an observer inside
        a core a radius of 0 for that core switches the test off for it.
    The cores are folded one at a time, so each pair stays in registers; M <= 64 keeps the
    unrolled loop small (compile and run time against M: see game-notes.md).
    """
    centre_x, centre_y, radius_squared = cores
    observer_x, observer_y, observer_in, alive, vision = observers
    target_x, target_y, target_in = targets
    line_x = target_x[..., :, None] - observer_x[..., None, :]  # (B, 2, Q, O): d
    line_y = target_y[..., :, None] - observer_y[..., None, :]
    length_squared = line_x * line_x + line_y * line_y
    blocked = (target_in[..., :, None] ^ observer_in[..., None, :]) != 0
    M = centre_x.shape[1]
    own_core = (observer_in[..., None] >> torch.arange(M, device=observer_in.device)) & 1 == 1
    radius_squared = torch.where(own_core, 0.0, radius_squared[:, None, None, :])  # (B, 2, O, M)
    for i in range(M):
        to_centre_x = centre_x[:, i, None, None, None] - observer_x[..., None, :]  # u
        to_centre_y = centre_y[:, i, None, None, None] - observer_y[..., None, :]
        along = to_centre_x * line_x + to_centre_y * line_y
        across = to_centre_x * line_y - to_centre_y * line_x
        blocked = blocked | ((across * across < radius_squared[..., None, :, i] * length_squared)
                             & (along > 0.0) & (along < length_squared))
    in_range = length_squared <= (vision * vision)[..., None, :]
    return (in_range & alive[..., None, :] & ~blocked).any(-1)


def vision_range(rules: Rules, state: State) -> torch.Tensor:
    """(B, N) each ship's vision range (m) as an observer. One shared value for now; a
    per-type range (the ship card) becomes a `per_ship` lookup here."""
    return torch.full_like(state.ships.x, rules.vision_range)


def visibility(rules: Rules, state: State) -> torch.Tensor:
    """(B, 2, N): ship n visible to team T, in sim order (game spec 10.1). Range tests use
    d^2 <= V^2 with the observer's V. In an environment without fog every living ship is
    visible."""
    fields, ships = state.fields, state.ships
    core = (fields.radius - 0.5 * fields.band_width).clamp_min(0.0)  # no flat interior: none
    cores = (fields.x, fields.y, core * core)
    observer_x, observer_y = by_team(ships.x), by_team(ships.y)
    target_x, target_y = enemy(ships.x), enemy(ships.y)
    observers = (observer_x, observer_y, core_membership(cores, observer_x, observer_y),
                 by_team(ships.alive), by_team(vision_range(rules, state)))
    seen = sighted(cores, observers, (target_x, target_y,
                                      core_membership(cores, target_x, target_y)))
    unfogged = ~state.match.fogged[:, None, None]
    enemies = enemy(ships.alive) & (enemy(ships.fired) | enemy(ships.spawned) | seen | unfogged)
    own = torch.ones_like(enemies)  # (B, 2, S) each; a team always sees its own ships
    return torch.stack([torch.cat([own[:, 0], enemies[:, 0]], -1),
                        torch.cat([enemies[:, 1], own[:, 1]], -1)], 1)
