"""The rewards and Frontline fixture interface over bnb (see README.md).

Each method is bnb's function under the fixture's name: the ledger and the levels'
events from `bnb.learn.rewards`, Frontline's rules from `bnb.game.modes.frontline`. The
translations are capture_step's: the fixture passes the living ships (`alive`), which the
game reads only through `inside` (already masked to them), and checks the majority's push
(`direction`), which the game does not return; the adapter computes it from the same
inputs, and the result is the fixture's dict.
"""

import torch

from bnb.game.modes import frontline
from bnb.learn import rewards


class RewardsArm:
    name = "bnb"

    accumulate_ledger = staticmethod(rewards.accumulate_ledger)
    attribute_recharge = staticmethod(rewards.attribute_recharge)
    enemy_team_share = staticmethod(rewards.enemy_team_share)
    outcome_events = staticmethod(rewards.outcome_events)
    kill_death_events = staticmethod(rewards.kill_death_events)
    damage_events = staticmethod(rewards.damage_events)

    roles_from_front = staticmethod(frontline.roles_from_front)
    terminal_distances = staticmethod(frontline.terminal_distances)
    harmonic = staticmethod(frontline.harmonic)
    zone_membership = staticmethod(frontline.zone_membership)
    zone_progress_events = staticmethod(frontline.zone_progress_events)
    timeout_result = staticmethod(frontline.timeout_result)

    @staticmethod
    def capture_step(team, alive, inside, roles, progress, attack_ledger, defense_ledger, front,
                     tick_seconds, capture_seconds):
        capture = frontline.capture_step(team, inside, roles, progress, attack_ledger,
                                         defense_ledger, front, tick_seconds, capture_seconds)
        out = capture._asdict()
        out["direction"] = _majority(team, inside, roles, capture)
        return out


def _majority(team, inside, roles, capture):
    """The majority's push on each defense zone (..., Z) in team 0's frame: +1 a team-0
    majority, -1 a team-1 one, 0 none, on a completion tick and off the defense zones."""
    present = inside.transpose(-1, -2)  # (..., Z, N)
    on_team0 = (team == 0)[..., None, :]
    net = (present & on_team0).sum(-1) - (present & ~on_team0).sum(-1)
    contested = (roles == frontline.TEAM0_DEFENSE) | (roles == frontline.TEAM1_DEFENSE)
    completed = (capture.team0_captured | capture.team1_captured)[..., None]
    return torch.where(contested & ~completed, torch.sign(net), 0)


def make_arm():
    return RewardsArm()
