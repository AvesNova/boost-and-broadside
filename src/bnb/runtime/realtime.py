"""One match on the CPU for interactive play: the same game functions at B = 1.

The tick is one compiled function (Inductor's C++ backend). Process-wide settings (thread
count) belong to the entry point.
"""

import torch

from bnb.contracts import GameConfig, Obs
from bnb.game import match
from bnb.game.perception.observe import perceive


class Realtime:
    def __init__(self, cfg: GameConfig, compile: bool = True) -> None:
        self.p = match.preset(cfg)
        self.state = match.reset(self.p, torch.rand(1, self.p.num_uniforms))
        self.state, self.obs, _ = perceive(self.p, self.state)
        self._tick = torch.compile(self._tick_fn, fullgraph=True, dynamic=False) if compile \
            else self._tick_fn

    def _tick_fn(self, state: dict, actions: torch.Tensor) -> tuple[dict, Obs]:
        p, N = self.p, actions.shape[0]
        moved, origin = match.move(p, state, actions[None], torch.randn(1, N, 2))
        fresh = match.new_map(p, torch.rand(1, p.num_uniforms))
        stepped, _ = match.resolve(p, moved, origin, fresh)
        stepped, obs, _ = perceive(p, stepped)
        return stepped, obs

    @torch.no_grad()
    def step(self, actions: torch.Tensor) -> Obs:
        """Advance one tick with (N, 3) int32 actions (sim order); returns every team's view,
        shaped (1, T, ...). Resets automatically at the end of an episode."""
        self.state, self.obs = self._tick(self.state, actions)
        return self.obs
