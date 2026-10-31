"""The runtime gives each controller its own team's view and applies its actions to that team."""

import torch

from bnb.contracts import Action, GameConfig, Obs, Outputs, Pool
from bnb.contracts.obs import SHIP
from bnb.runtime.rollout import Rollout

CPU = torch.device("cpu")


class Constant:
    """Always the same power setting; records the own-row speeds it was shown."""

    reads = ("ships",)

    def __init__(self, power: int, ships_per_team: int) -> None:
        self.power, self.S = power, ships_per_team
        self.seen: list[torch.Tensor] = []

    def init_carry(self, views, device):
        return None

    def step(self, carry, obs: Obs, done):
        self.seen.append(obs.ships[:, : self.S, SHIP["w"]].clone())
        shape = obs.ships.shape[:1] + (self.S,)
        zero = torch.zeros(shape, dtype=torch.int32)
        return Action(zero + self.power, zero, zero), carry, Outputs()


def test_actions_and_views_follow_the_seat():
    cfg = GameConfig(ships_per_team=3, num_fields=2, arena_radius=5000.0)
    fast, slow = Constant(1, 3), Constant(2, 3)  # boost vs reverse
    pools = (Pool("a", 0.5, ("fast", "slow")), Pool("b", 0.5, ("slow", "fast")))
    ro = Rollout(cfg, pools, {"fast": fast, "slow": slow}, "fast", 8, 40, CPU, compile=False)
    ro.run()
    st, seat = ro.carry["state"], ro.carry["seat"]
    w = st["w"].reshape(8, 2, 3).mean(-1)  # (B, team)
    lineup = ro.lineup
    for name, quick in (("fast", True), ("slow", False)):
        env = lineup.env[name]
        team = (lineup.role[name] + seat[env]) % 2
        mine, theirs = w[env, team], w[env, 1 - team]
        assert bool(((mine > theirs) == quick).all()), (name, mine, theirs)
    # The fast controller was shown fast own ships (its view is its own team).
    assert fast.seen[-1].mean() > slow.seen[-1].mean()
