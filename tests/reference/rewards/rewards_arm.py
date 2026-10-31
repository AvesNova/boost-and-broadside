"""The rewards and Frontline fixture interface over bnb (see README.md).

Each method is bnb's function under the fixture's name: the ledger and the levels'
events from `bnb.learn.rewards`, Frontline's rules from `bnb.game.modes.frontline`. The
only translation is capture_step's named result to the fixture's dict.
"""

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
    def capture_step(*args):
        return frontline.capture_step(*args)._asdict()


def make_arm():
    return RewardsArm()
