"""Visibility (game spec 10.1): which ships each team sees. Bullets are not perceived."""

from typing import NamedTuple

import torch

from ..rules import Rules
from ..state import State, by_team


def enemy(v: torch.Tensor, teams: int = 2) -> torch.Tensor:
    """(B, N) per-ship values as seen by each team's enemy: row T holds team 1 - T."""
    return by_team(v, teams).flip(1)


def core_membership(cores: tuple, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """(B, T, P) int64 points as bitmasks: bit i is set when the point is strictly inside
    core i. With 64 cores bit 63 is the sign bit; XOR, the shifts below and this sum of
    disjoint bits all still read it correctly."""
    cx, cy, rho2 = cores
    M = cx.shape[1]
    dx = x[..., None] - cx[:, None, None, :]
    dy = y[..., None] - cy[:, None, None, :]
    inside = dx * dx + dy * dy < rho2[:, None, None, :]
    return (inside.long() << torch.arange(M, device=x.device)).sum(-1)


def sighted(cores: tuple, observers: tuple, targets: tuple) -> torch.Tensor:
    """(B, T, Q): some living observer of team T is within its range of target q with a
    clear sight line (game spec 10.1).

    cores: centre x, y and radius^2, (B, M); observers: (x, y, membership, alive, range),
    (B, T, O); targets: (x, y, membership), (B, T, Q); membership from `core_membership`.

    A segment comes within rho of a core centre exactly when an end is inside the core, or
    the foot of the perpendicular from the centre falls inside the segment (0 < u.d < |d|^2,
    with u = centre - observer, d = target - observer) at a distance below rho
    (|u x d|^2 < rho^2 |d|^2). With the exception for two ends inside the same core, a
    core blocks the line when
      - exactly one end is inside it: the bitmasks differ in its bit, or
      - neither end is inside and the perpendicular test holds. Giving an observer inside
        a core a radius of 0 for that core switches the test off for it.
    The cores are folded one at a time, so each pair stays in registers (M <= 64 keeps the
    unrolled loop small).
    """
    cx, cy, rho2 = cores
    ox, oy, o_in, alive, reach = observers
    tx, ty, t_in = targets
    dx = tx[..., :, None] - ox[..., None, :]  # (B, T, Q, O), observer to target
    dy = ty[..., :, None] - oy[..., None, :]
    dd = dx * dx + dy * dy
    blocked = (t_in[..., :, None] ^ o_in[..., None, :]) != 0
    M = cx.shape[1]
    own_core = (o_in[..., None] >> torch.arange(M, device=o_in.device)) & 1 == 1  # (B, T, O, M)
    rho2 = torch.where(own_core, 0.0, rho2[:, None, None, :])
    for i in range(M):
        ux = cx[:, i, None, None, None] - ox[..., None, :]  # observer to core centre
        uy = cy[:, i, None, None, None] - oy[..., None, :]
        along = ux * dx + uy * dy
        across = ux * dy - uy * dx
        blocked = blocked | ((across * across < rho2[..., None, :, i] * dd) & (along > 0.0)
                             & (along < dd))
    in_range = dd <= (reach * reach)[..., None, :]
    return (in_range & alive[..., None, :] & ~blocked).any(-1)


class Visibility(NamedTuple):
    ships: torch.Tensor  # (B, T, N): ship n visible to team T
    enemy_ships: torch.Tensor  # (B, T, S): team 1 - T's ships seen by team T


def vision_range(rules: Rules, st: State) -> torch.Tensor:
    """(B, N) each ship's vision range (m) as an observer. One shared V for now; a per-type
    range (the ship card) is a `per_ship` lookup here."""
    return torch.full_like(st.ships.x, rules.vision)


def visibility(rules: Rules, st: State) -> Visibility:
    """Who sees what (game spec 10.1), for both teams. Range tests use d^2 <= V^2 with the
    observer's V."""
    f, sh, T = st.fields, st.ships, rules.num_teams
    rho = (f.radius - 0.5 * f.width).clamp_min(0.0)  # a field with no flat interior: no core
    cores = (f.x, f.y, rho * rho)
    own, foe = (lambda v: by_team(v, T)), (lambda v: enemy(v, T))
    ox, oy, tx, ty = own(sh.x), own(sh.y), foe(sh.x), foe(sh.y)
    observers = (ox, oy, core_membership(cores, ox, oy), own(sh.alive),
                 own(vision_range(rules, st)))
    seen = sighted(cores, observers, (tx, ty, core_membership(cores, tx, ty)))
    enemy_ships = foe(sh.alive) & (foe(sh.fired) | foe(sh.spawned) | seen)  # (B, T, S)
    # Each team's row over every ship in sim order (own ships always visible); T = 2.
    mine = torch.ones_like(enemy_ships)
    ships = torch.stack([torch.cat([mine[:, 0], enemy_ships[:, 0]], -1),
                         torch.cat([enemy_ships[:, 1], mine[:, 1]], -1)], 1)
    return Visibility(ships, enemy_ships)
