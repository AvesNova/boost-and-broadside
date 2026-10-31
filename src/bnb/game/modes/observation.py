"""What an agent may know of the mode (modes spec 7), team-relative like the rest of the
observation: the zone rows and the mode's global columns, per view (view v is team v)."""

import torch

from bnb.contracts.obs import GLOBAL_COLUMNS, ZONE_COLUMNS

from ..rules import TEAMS, Rules
from ..state import ModeState
from . import frontline
from .rules import FRONTLINE, TEAM_DEATHMATCH

MODE_GLOBALS = GLOBAL_COLUMNS[GLOBAL_COLUMNS.index("mode_team_deathmatch"):]


def zone_rows(rules: Rules, mode: ModeState) -> torch.Tensor:
    """(B, 2, Z, len(ZONE_COLUMNS)) in ZONE_COLUMNS order (Z = 0 outside Frontline)."""
    role = mode.role
    B, Z = role.shape
    if Z == 0:
        return torch.zeros((B, TEAMS, 0, len(ZONE_COLUMNS)), device=role.device)
    offensive, defensive = frontline.terminal_distances(mode.front, rules.mode.front_threshold)
    team0_push = mode.direction  # the majority's push, in team 0's frame
    views = []
    for relative_role, push, own, enemy in ((role, team0_push, offensive, defensive),
                                            (frontline.TEAM1_SPAWN - role, -team0_push,
                                             defensive, offensive)):
        views.append(torch.stack([mode.zone_x, mode.zone_y, mode.zone_radius,
                                  relative_role.float(), mode.progress, push.float(),
                                  own.float(), enemy.float()], -1))
    return torch.stack(views, 1)


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
