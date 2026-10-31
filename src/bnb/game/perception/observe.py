"""The observation contract (bnb.contracts.obs) from the game state: ships, fields, the
mode's zones and the globals (the mode's columns last).

Each view lists rows by team_relative: view v's row block j holds team (j + v) mod 2. With
team-major storage this is a static gather, fused into the copy this stage writes anyway.
"""

import torch

from bnb.contracts import Obs, Privileged
from bnb.contracts.obs import SHIP_COLUMNS, SHIP_HIDDEN_ZERO

from ..modes.observation import mode_globals, zone_rows
from ..rules import TEAMS, Rules
from ..state import State, replace
from .sight import visibility

# The row layout `ship_rows` writes: the hidden columns, then the always-shown ones.
HIDDEN = ("x", "y", "heading_cos", "heading_sin", "velocity_x", "velocity_y", "proper_speed",
          "angle_of_attack", "nose_rate", "energy", "shield", "shield_recharge_wait",
          "gun_cooldown", "log_refractive_index", "log_index_gradient_x", "log_index_gradient_y")
SHOWN = ("alive", "visible", "fired", "respawn_wait", "ship_type", "team_relative",
         "time_since_seen", "spawned")
assert HIDDEN + SHOWN == SHIP_COLUMNS and len(HIDDEN) == SHIP_HIDDEN_ZERO

TICKS_SINCE_SEEN_CAP = 1 << 30  # keeps the counter far from int32 overflow


def view_order(ships_per_team: int, device) -> torch.Tensor:
    """(2, N) int64: the sim index of each row of each view."""
    slot = torch.arange(ships_per_team, device=device)
    rows = [torch.cat([((j + v) % TEAMS) * ships_per_team + slot for j in range(TEAMS)])
            for v in range(TEAMS)]
    return torch.stack(rows)


def _hidden_columns(rules: Rules, state: State) -> torch.Tensor:
    """(B, N, 16) the columns a team sees only for ships it can see (sim order)."""
    ships = state.ships
    speed = ships.proper_speed / torch.exp(ships.log_refractive_index)
    cos, sin = ships.heading_cos, ships.heading_sin
    angle_cos, angle_sin = ships.angle_of_attack_cos, ships.angle_of_attack_sin
    velocity_x = speed * (cos * angle_cos + sin * angle_sin)
    velocity_y = speed * (sin * angle_cos - cos * angle_sin)
    return torch.stack([ships.x, ships.y, cos, sin, velocity_x, velocity_y, ships.proper_speed,
                        ships.angle_of_attack, ships.nose_rate, ships.energy, ships.shield,
                        ships.shield_recharge_wait, ships.gun_cooldown.float() * rules.tick_length,
                        ships.log_refractive_index, ships.log_index_gradient_x,
                        ships.log_index_gradient_y], -1)


def perceive(rules: Rules, state: State) -> tuple[State, Obs, Privileged]:
    """Visibility, the per-view observation and the truth. Returns the state with
    `ticks_since_seen` advanced (the only state perception changes)."""
    ships, fields, match = state.ships, state.fields, state.match
    B, N = ships.x.shape
    S, device = rules.ships_per_team, ships.x.device
    order = view_order(S, device)  # (2, N)
    views = torch.arange(TEAMS, device=device)[:, None]

    visible = visibility(rules, state)  # (B, 2, N) sim order
    unfogged = ~match.fogged[:, None, None]
    since = torch.where(visible | unfogged, 0,
                        (match.ticks_since_seen + 1).clamp_max(TICKS_SINCE_SEEN_CAP))

    hidden = _hidden_columns(rules, state)[:, order]  # (B, 2, N, 16) by row
    team_relative = (torch.arange(N, device=device) // S).float()  # by row: static
    alive, fired = ships.alive[:, order].float(), ships.fired[:, order].float()
    respawn_wait = ships.respawn_wait[:, order].float() * rules.tick_length
    ship_type, spawned = ships.ship_type[:, order].float(), ships.spawned[:, order].float()

    def ship_rows(seen: torch.Tensor, ticks: torch.Tensor) -> torch.Tensor:
        """seen, ticks: (B, 2, N) in sim order."""
        seen, ticks = seen[:, views, order], ticks[:, views, order]
        shown = torch.stack([alive, seen.float(), fired, respawn_wait, ship_type,
                             team_relative.expand(B, TEAMS, N), ticks.float() * rules.tick_length,
                             spawned], -1)
        return torch.cat([torch.where(seen[..., None], hidden, 0.0), shown], -1)

    field_rows = torch.stack([fields.x, fields.y, fields.radius, fields.band_width,
                              fields.level.float()], -1)
    tick = match.episode_tick.float()
    global_row = torch.stack([tick / rules.episode_ticks, torch.full_like(tick, rules.arena_radius),
                              match.fogged.float(), (match.episode_tick == 0).float()], -1)
    globals_ = torch.cat([global_row[:, None].expand(B, TEAMS, -1),
                          mode_globals(rules, state.mode)], -1)
    obs = Obs(ships=ship_rows(visible, since), fields=field_rows[:, None].expand(B, TEAMS, -1, -1),
              globals=globals_, zones=zone_rows(rules, state.mode))

    truth = Privileged(team=(torch.arange(N, device=device) // S).to(torch.int32).expand(B, N),
                       view_order=order.expand(B, TEAMS, N),
                       ships=ship_rows(torch.ones_like(visible), torch.zeros_like(since)))
    return replace(state, match=replace(match, ticks_since_seen=since)), obs, truth
