"""One match on the CPU for interactive play: the same game functions at B = 1.

The tick is one compiled function (Inductor's C++ backend). Process-wide settings (thread
count) belong to the entry point.
"""

import torch

from bnb.contracts import Action, GameConfig, Obs
from bnb.game import api


class Realtime:
    def __init__(self, cfg: GameConfig, compile: bool = True, fogged: bool = True) -> None:
        self.rules = api.rules_of(cfg)
        self.types = api.ship_types()
        fog = torch.tensor([fogged])
        self.state = api.reset(self.rules, self.types, api.uniforms(self.rules, 1), fog)
        self.state, self.obs, _ = api.perceive(self.rules, self.state)
        self._tick = torch.compile(self._tick_fn, fullgraph=True, dynamic=False) if compile \
            else self._tick_fn

    def _tick_fn(self, state: api.State, types: api.ShipTypes,
                 actions: torch.Tensor) -> tuple[api.State, Obs]:
        rules = self.rules
        moved, sweep = api.move(rules, types, state, Action.unpack(actions[None]),
                                api.spread(rules, 1))
        stepped, _ = api.resolve(rules, types, moved, sweep, api.uniforms(rules, 1))
        stepped, obs, _ = api.perceive(rules, stepped)
        return stepped, obs

    @torch.no_grad()
    def step(self, actions: torch.Tensor) -> Obs:
        """Advance one tick with (N, 3) int32 actions (sim order); returns every team's view,
        shaped (1, 2, ...). Resets automatically at the end of an episode."""
        self.state, self.obs = self._tick(self.state, self.types, actions)
        return self.obs
