"""What an agent may know of the mode (modes spec 7), team-relative like the rest of the
observation: the zone rows and the mode's global columns, per view (view v is team v).

The ring is a mirror image about the neutral zone, so team 1 lists the zones in mirrored
cyclic order, zone (-z) mod 5 in row z: at front 0 both views read neutral, own spawn, own
defense, enemy defense, enemy spawn, and nothing in the rows reveals the seat."""

import torch

from bnb.contracts.obs import GLOBAL_COLUMNS, ZONE_COLUMNS

from ..rules import TEAMS, Rules
from ..state import ModeState
from . import frontline
from .rules import FRONTLINE, TEAM_DEATHMATCH

MODE_GLOBALS = GLOBAL_COLUMNS[GLOBAL_COLUMNS.index("mode_team_deathmatch"):]


def zone_rows(rules: Rules, mode: ModeState) -> torch.Tensor:
    """(B, 2, Z, len(ZONE_COLUMNS)) in ZONE_COLUMNS order (Z = 0 outside Frontline).

    direction: the meter's movement this tick, +1 toward the viewer's capture (a meter
    rising on the zone the viewer attacks, or falling on the one it defends), -1 toward
    the enemy's, 0 still."""
    role = mode.role
    B, Z = role.shape
    if Z == 0:
        return torch.zeros((B, TEAMS, 0, len(ZONE_COLUMNS)), device=role.device)
    offensive, defensive = frontline.terminal_distances(mode.front, rules.mode.front_threshold)
    team0_attacks = torch.where(role == frontline.TEAM1_DEFENSE, 1,
                                torch.where(role == frontline.TEAM0_DEFENSE, -1, 0))
    team0_direction = mode.movement * team0_attacks  # in team 0's favour
    team0 = torch.stack([mode.zone_x, mode.zone_y, mode.zone_radius, role.float(),
                         mode.progress, team0_direction.float(), offensive.float(),
                         defensive.float()], -1)
    team1 = torch.stack([mode.zone_x, mode.zone_y, mode.zone_radius,
                         (frontline.TEAM1_SPAWN - role).float(), mode.progress,
                         -team0_direction.float(), defensive.float(), offensive.float()], -1)
    mirrored = (-torch.arange(Z, device=role.device)) % Z
    return torch.stack([team0, team1[:, mirrored]], 1)


def mode_globals(rules: Rules, mode: ModeState) -> torch.Tensor:
    """(B, 2, len(MODE_GLOBALS)): the mode one-hot, the scores by team_relative and the
    winning score, the front (positive for the viewer) and the winning front."""
    B = mode.front.shape[0]
    zero = torch.zeros(B, device=mode.front.device)
    name = rules.mode.name
    score = mode.score.float()
    target = rules.mode.kills_per_ship * rules.ships_per_team if name == TEAM_DEATHMATCH else 0
    threshold = rules.mode.front_threshold if name == FRONTLINE else 0
    front = mode.front.float()
    views = [torch.stack([zero + float(name == TEAM_DEATHMATCH), zero + float(name == FRONTLINE),
                          score[:, view], score[:, 1 - view], zero + float(target),
                          front if view == 0 else -front, zero + float(threshold)], -1)
             for view in range(TEAMS)]
    return torch.stack(views, 1)
