"""Visibility (race SPEC 10.1): which ships each team sees. Bullets are not observed."""

from typing import NamedTuple

import torch

from ..rules import Rules
from ..state import State, by_team


def enemy(v: torch.Tensor) -> torch.Tensor:
    """(B, N) per-ship values as seen by each team's enemy: row T holds team 1 - T."""
    return by_team(v).flip(1)


def core_membership(cores: dict, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """(B, 2, P) int64 points as bitmasks: bit i is set when the point is strictly inside core i."""
    M = cores["fx"].shape[1]
    assert M <= 64, "core membership is packed into an int64 bitmask (SPEC 1.2: M <= 64)"
    # With 64 fields, bit 63 is the sign bit: XOR, the arithmetic shift-and-mask in
    # `sighted` and this sum of disjoint bits all still read it correctly.
    dx = x[..., None] - cores["fx"][:, None, None, :]
    dy = y[..., None] - cores["fy"][:, None, None, :]
    inside = dx * dx + dy * dy < cores["rho2"][:, None, None, :]
    return (inside.long() << torch.arange(M, device=x.device)).sum(-1)


def sighted(p: Rules, cores: dict, observers: tuple, targets: tuple) -> torch.Tensor:
    """(B, 2, Q): some living observer of the team is within V of the target with a clear line.

    observers: (x, y, alive, membership) per team, (B, 2, O); targets: (x, y, membership),
    (B, 2, Q), membership from `core_membership`.

    A segment comes within rho of a core center exactly when an end is inside the core,
    or the foot of the perpendicular from the center falls inside the segment
    (0 < u.d < |d|^2, with u = center - observer, d = target - observer) and the
    perpendicular distance is below rho (|u x d|^2 < rho^2 |d|^2). With the exception
    for two ends inside the same core, a core blocks the line when
      - exactly one end is inside it: the bitmasks differ in its bit, or
      - neither end is inside and the perpendicular test holds. Giving an observer inside
        a core a radius of 0 for that core switches the test off for it.
    The fields are folded one at a time, so each pair stays in registers.
    """
    ox, oy, alive, o_in = observers
    tx, ty, t_in = targets
    dx = tx[..., :, None] - ox[..., None, :]  # (B, 2, Q, O), observer to target
    dy = ty[..., :, None] - oy[..., None, :]
    dd = dx * dx + dy * dy
    blocked = (t_in[..., :, None] ^ o_in[..., None, :]) != 0
    M = cores["fx"].shape[1]
    own_core = (o_in[..., None] >> torch.arange(M, device=o_in.device)) & 1 == 1  # (B, 2, O, M)
    rho2 = torch.where(own_core, 0.0, cores["rho2"][:, None, None, :])
    for i in range(M):
        ux = cores["fx"][:, i, None, None, None] - ox[..., None, :]  # observer to core center
        uy = cores["fy"][:, i, None, None, None] - oy[..., None, :]
        along = ux * dx + uy * dy
        across = ux * dy - uy * dx
        blocked = blocked | ((across * across < rho2[..., None, :, i] * dd) & (along > 0.0) & (along < dd))
    return ((dd <= p.vision ** 2) & alive[..., None, :] & ~blocked).any(-1)


class Visibility(NamedTuple):
    """Per looking team T (the team axis holds the teams asked for, both by default)."""

    ships: torch.Tensor  # (B, T, N): ship k visible to team T
    enemy_ships: torch.Tensor  # (B, T, N/2): team 1 - T's ships seen by team T


def visibility(p: Rules, st: State, teams: slice = slice(None)) -> Visibility:
    """Who sees what (SPEC 10.1). teams: the looking teams, as a slice of the team axis."""
    # Opaque cores: radius max(R - b/2, 0); a field too small for a flat interior has none.
    f, sh = st.fields, st.ships
    rho = (f.radius - 0.5 * f.width).clamp_min(0.0)
    cores = dict(fx=f.x, fy=f.y, rho2=rho * rho)

    def own(v):
        return by_team(v)[:, teams]

    def foreign(v):
        return enemy(v)[:, teams]

    def located(x, y):  # (x, y, core membership), each (B, T, ...)
        return x, y, core_membership(cores, x, y)

    ox, oy, o_in = located(own(sh.x), own(sh.y))
    observers = (ox, oy, own(sh.alive), o_in)
    seen_ships = sighted(p, cores, observers, located(foreign(sh.x), foreign(sh.y)))
    enemy_ships = foreign(sh.alive) & (foreign(sh.fired) | foreign(sh.spawned) | seen_ships)

    def all_ships(team, own_part, enemy_part):  # one team's row over every ship, team 0's first
        return torch.cat([own_part, enemy_part] if team == 0 else [enemy_part, own_part], -1)

    looking = range(2)[teams]
    ships = torch.stack([all_ships(T, torch.ones_like(enemy_ships[:, i]), enemy_ships[:, i])
                         for i, T in enumerate(looking)], 1)
    return Visibility(ships, enemy_ships)
