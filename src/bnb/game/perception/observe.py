"""The observation contract (bnb.contracts.obs) from the game state.

Each view lists rows by team_rel: view v's row block j holds team (j + v) mod T. With
team-major storage this is a static gather, fused into the copy this stage writes anyway.
"""

import torch

from bnb.contracts import Obs, Privileged
from bnb.contracts.obs import SHIP_HIDDEN_ZERO

from ..physics.constants import Preset
from .sight import visibility

LAST_SEEN_CAP = 1 << 30  # ticks; keeps the counter far from int32 overflow


def view_order(num_teams: int, ships_per_team: int, device) -> torch.Tensor:
    """(T, N) int64: the sim index of each row of each view."""
    slot = torch.arange(ships_per_team, device=device)
    rows = [torch.cat([((j + v) % num_teams) * ships_per_team + slot for j in range(num_teams)])
            for v in range(num_teams)]
    return torch.stack(rows)


def _ship_columns(p: Preset, st: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, N, 12) hidden-when-unseen columns and (B, N, 3) always-shown ones (sim order)."""
    n = torch.exp(st["logn"])
    c, s, ca, sa, w = st["c"], st["s"], st["ca"], st["sa"], st["w"]
    vx, vy = (w / n) * (c * ca + s * sa), (w / n) * (s * ca - c * sa)
    hidden = torch.stack([st["x"], st["y"], c, s, vx, vy, w, st["alpha"], st["r"], st["E"],
                          st["shield"], st["gun_wait"].float() * p.h], -1)
    shown = torch.stack([st["alive"].float(), st["respawn"].float() * p.h,
                         st["ship_type"].float()], -1)
    return hidden, shown


def perceive(p: Preset, st: dict) -> tuple[dict, Obs, Privileged]:
    """Visibility, the per-view observation and the truth. Returns the state with
    `last_seen` advanced (the only state perception changes)."""
    B, N = st["x"].shape
    T, S, K = 2, p.half, p.slots
    device = st["x"].device
    order = view_order(T, S, device)  # (T, N)
    views = torch.arange(T, device=device)[:, None]

    vis = visibility(p, st)
    last_seen = torch.where(vis.ships, 0, (st["last_seen"] + 1).clamp_max(LAST_SEEN_CAP))

    hidden, shown = _ship_columns(p, st)
    team_rel = (torch.arange(N, device=device) // S).float()  # by row: the ordering makes it static

    def ship_rows(seen: torch.Tensor, since: torch.Tensor) -> torch.Tensor:
        masked = torch.where(seen[..., None], hidden[:, None], 0.0)  # (B, T, N, 12)
        rows = torch.cat([masked, shown[:, None].expand(B, T, N, 3)], -1)[:, views, order]
        return torch.cat([rows[..., :SHIP_HIDDEN_ZERO + 1], seen[:, views, order, None].float(),
                          rows[..., SHIP_HIDDEN_ZERO + 1:], team_rel.expand(B, T, N)[..., None],
                          since[:, views, order, None], st["spawned"][:, order, None].float()], -1)

    ships = ship_rows(vis.ships, last_seen.float() * p.h)

    bn = torch.exp(st["blogn"]).reshape(B, N * K)
    bw = st["bw"].reshape(B, N * K)
    bcols = torch.stack([st["bx"].reshape(B, -1), st["by"].reshape(B, -1),
                         st["bdx"].reshape(B, -1) * bw / bn, st["bdy"].reshape(B, -1) * bw / bn], -1)
    border = (order[..., None] * K + torch.arange(K, device=device)).reshape(T, N * K)
    bvis = vis.bullets[:, views, border]  # (B, T, N K)
    bullets = torch.cat([torch.where(bvis[..., None], bcols[:, border], 0.0), bvis[..., None].float(),
                         torch.where(bvis, (torch.arange(N * K, device=device) // (S * K)).float(),
                                     0.0)[..., None]], -1)

    fields = torch.stack([st["fx"], st["fy"], st["fr"], st["flevel"].float()], -1)
    t = st["t"].float()
    glob = torch.stack([t / p.episode_ticks, torch.full_like(t, p.arena_radius), torch.ones_like(t),
                        (st["t"] == 0).float()], -1)
    obs = Obs(ships=ships, bullets=bullets, fields=fields[:, None].expand(B, T, -1, -1),
              globals=glob[:, None].expand(B, T, -1))

    everything = torch.ones_like(vis.ships)
    truth = Privileged(team=(torch.arange(N, device=device) // S).to(torch.int32).expand(B, N),
                       view_order=order.expand(B, T, N),
                       ships=ship_rows(everything, torch.zeros_like(last_seen, dtype=torch.float32)))
    return dict(st, last_seen=last_seen), obs, truth
