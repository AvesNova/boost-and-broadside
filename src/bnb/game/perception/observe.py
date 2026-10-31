"""The observation contract (bnb.contracts.obs) from the game state.

Each view lists rows by team_rel: view v's row block j holds team (j + v) mod T. With
team-major storage this is a static gather, fused into the copy this stage writes anyway.
"""

import torch

from bnb.contracts import Obs, Privileged
from bnb.contracts.obs import SHIP_COLUMNS, SHIP_HIDDEN_ZERO

from ..rules import Rules
from ..state import State, replace
from .sight import visibility

# The row layout `ship_rows` writes: the hidden columns, then the always-shown ones.
HIDDEN = ("x", "y", "c", "s", "vx", "vy", "w", "alpha", "r", "E", "shield", "gun_wait")
SHOWN = ("alive", "visible", "respawn", "type", "team_rel", "since_seen", "spawned")
assert HIDDEN + SHOWN == SHIP_COLUMNS and len(HIDDEN) == SHIP_HIDDEN_ZERO

LAST_SEEN_CAP = 1 << 30  # ticks; keeps the counter far from int32 overflow


def view_order(num_teams: int, ships_per_team: int, device) -> torch.Tensor:
    """(T, N) int64: the sim index of each row of each view."""
    slot = torch.arange(ships_per_team, device=device)
    rows = [torch.cat([((j + v) % num_teams) * ships_per_team + slot for j in range(num_teams)])
            for v in range(num_teams)]
    return torch.stack(rows)


def _ship_columns(p: Rules, st: State) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, N, 12) hidden-when-unseen columns and (B, N, 3) always-shown ones (sim order)."""
    sh = st.ships
    n = torch.exp(sh.logn)
    c, s, ca, sa, w = sh.c, sh.s, sh.ca, sh.sa, sh.w
    vx, vy = (w / n) * (c * ca + s * sa), (w / n) * (s * ca - c * sa)
    hidden = torch.stack([sh.x, sh.y, c, s, vx, vy, w, sh.alpha, sh.r, sh.E,
                          sh.shield, sh.gun_wait.float() * p.tick], -1)
    shown = torch.stack([sh.alive.float(), sh.respawn.float() * p.tick,
                         sh.ship_type.float()], -1)
    return hidden, shown


def perceive(p: Rules, st: State) -> tuple[State, Obs, Privileged]:
    """Visibility, the per-view observation and the truth. Returns the state with
    `last_seen` advanced (the only state perception changes)."""
    B, N = st.ships.x.shape
    T, S = p.num_teams, p.ships_per_team
    device = st.ships.x.device
    order = view_order(T, S, device)  # (T, N)
    views = torch.arange(T, device=device)[:, None]

    vis = visibility(p, st)
    last_seen = torch.where(vis.ships, 0, (st.match.last_seen + 1).clamp_max(LAST_SEEN_CAP))

    hidden, shown = (v[:, order] for v in _ship_columns(p, st))  # (B, T, N, ...) by row
    team_rel = (torch.arange(N, device=device) // S).float()  # by row: the ordering makes it static
    extra = torch.stack([team_rel.expand(B, T, N), st.ships.spawned[:, order].float()], -1)

    def ship_rows(seen: torch.Tensor, since: torch.Tensor) -> torch.Tensor:
        """seen, since: (B, T, N) in sim order."""
        seen, since = seen[:, views, order, None], since[:, views, order, None]
        alive, rest = shown[..., :1], shown[..., 1:]  # alive, then respawn and type
        return torch.cat([torch.where(seen, hidden, 0.0), alive, seen.float(), rest,
                          extra[..., :1], since, extra[..., 1:]], -1)

    ships = ship_rows(vis.ships, last_seen.float() * p.tick)

    f = st.fields
    fields = torch.stack([f.x, f.y, f.radius, f.level.float()], -1)
    t = st.match.t.float()
    glob = torch.stack([t / p.episode_ticks, torch.full_like(t, p.arena_radius), torch.ones_like(t),
                        (st.match.t == 0).float()], -1)
    obs = Obs(ships=ships, fields=fields[:, None].expand(B, T, -1, -1),
              globals=glob[:, None].expand(B, T, -1))

    everything, never = torch.ones_like(vis.ships), torch.zeros_like(last_seen, dtype=torch.float32)
    truth = Privileged(team=(torch.arange(N, device=device) // S).to(torch.int32).expand(B, N),
                       view_order=order.expand(B, T, N), ships=ship_rows(everything, never))
    return replace(st, match=replace(st.match, last_seen=last_seen)), obs, truth
