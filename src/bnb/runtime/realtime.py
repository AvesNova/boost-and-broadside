"""One match on the CPU for interactive play: the same game functions at B = 1.

The tick is one compiled function (Inductor's C++ backend). Process-wide settings (thread
count) belong to the entry point.
"""

import torch

from bnb.contracts import Action, GameConfig, Obs
from bnb.game import match


class Realtime:
    def __init__(self, cfg: GameConfig, compile: bool = True) -> None:
        self.p = match.rules_of(cfg)
        self.types = match.ship_types()
        self.state = match.reset(self.p, match.uniforms(self.p, 1), self.types)
        self.state, self.obs, _ = match.perceive(self.p, self.state)
        self._tick = torch.compile(self._tick_fn, fullgraph=True, dynamic=False) if compile \
            else self._tick_fn

    def _tick_fn(self, state: match.State, types: match.ShipTypes,
                 actions: torch.Tensor) -> tuple[match.State, Obs]:
        p = self.p
        moved, origin = match.move(p, state, Action.unpack(actions[None]), match.spread(p, 1),
                                   types)
        stepped, _ = match.resolve(p, moved, origin, match.uniforms(p, 1), types)
        stepped, obs, _ = match.perceive(p, stepped)
        return stepped, obs

    @torch.no_grad()
    def step(self, actions: torch.Tensor) -> Obs:
        """Advance one tick with (N, 3) int32 actions (sim order); returns every team's view,
        shaped (1, T, ...). Resets automatically at the end of an episode."""
        self.state, self.obs = self._tick(self.state, self.types, actions)
        return self.obs
