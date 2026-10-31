"""The skeleton's baseline model: a small MLP over one team's view.

Each own ship reads its own row, every ship's position and velocity relative to it in its
nose frame (so bearings need no learning), and a summary of the whole view (every ship
row, the fields and the globals). The first layer is split along that concatenation, so the summary
half is computed once per view instead of once per ship. The model reads token position
(the summary is a flattened view), so it relies on rows being ordered by team_rel.
"""

import torch
import torch.nn as nn

from bnb.contracts import Obs
from bnb.contracts.obs import FIELD_COLUMNS, GLOBAL_COLUMNS, SHIP, SHIP_COLUMNS

from .heads import SIZES

SHIP_SCALE = {"x": 1e-3, "y": 1e-3, "vx": 1e-2, "vy": 1e-2, "w": 1e-2, "E": 1e-4,
              "shield": 1e-2, "respawn": 0.125}
FIELD_SCALE = (1e-3, 1e-3, 1e-2, 1.0)
EGO = 5  # per other ship: forward, left (1e-3 m), closing velocity forward, left (1e-2 m/s), seen
GLOBAL_SCALE = (1.0, 1e-3, 1.0, 1.0)


class MLP(nn.Module):
    reads = ("ships", "fields", "globals")

    def __init__(self, ships_per_team: int, num_ships: int, num_fields: int, hidden: int = 256):
        super().__init__()
        C = len(SHIP_COLUMNS)
        self.ships_per_team = ships_per_team
        summary = num_ships * C + num_fields * len(FIELD_COLUMNS) + len(GLOBAL_COLUMNS)
        self.own = nn.Linear(C + EGO * num_ships, hidden)
        self.summary = nn.Linear(summary, hidden, bias=False)
        self.body = nn.Linear(hidden, hidden)
        self.policy = nn.Linear(hidden, sum(SIZES))
        self.value = nn.Linear(hidden, 1)
        scale = torch.tensor([SHIP_SCALE.get(name, 1.0) for name in SHIP_COLUMNS])
        self.register_buffer("ship_scale", scale, persistent=False)
        self.register_buffer("field_scale", torch.tensor(FIELD_SCALE), persistent=False)
        self.register_buffer("global_scale", torch.tensor(GLOBAL_SCALE), persistent=False)
        nn.init.zeros_(self.policy.weight)  # a uniform policy at the start
        nn.init.zeros_(self.policy.bias)

    def forward(self, obs: Obs) -> tuple[torch.Tensor, torch.Tensor]:
        """Logits (..., S, 12) and values (..., S) for views with any leading axes."""
        ships = obs.ships * self.ship_scale
        since = SHIP["since_seen"]
        ships = torch.cat([ships[..., :since], torch.log1p(ships[..., since:since + 1]),
                           ships[..., since + 1:]], -1)
        summary = torch.cat([ships.flatten(-2), (obs.fields * self.field_scale).flatten(-2),
                             obs.globals * self.global_scale], -1)
        own = ships[..., : self.ships_per_team, :]
        h = torch.relu(self.own(torch.cat([own, self._egocentric(obs.ships)], -1))
                       + self.summary(summary)[..., None, :])
        h = torch.relu(self.body(h))
        return self.policy(h), self.value(h)[..., 0]

    def _egocentric(self, ships: torch.Tensor) -> torch.Tensor:
        """(..., S, N * EGO): every ship as seen from each own ship's nose frame."""
        S = self.ships_per_team
        col = {k: ships[..., SHIP[k]] for k in ("x", "y", "vx", "vy", "c", "s", "visible")}
        own = {k: v[..., :S, None] for k, v in col.items()}
        dx, dy = col["x"][..., None, :] - own["x"], col["y"][..., None, :] - own["y"]
        dvx, dvy = col["vx"][..., None, :] - own["vx"], col["vy"][..., None, :] - own["vy"]
        c, s = own["c"], own["s"]
        seen = col["visible"][..., None, :].expand_as(dx)
        ego = torch.stack([(c * dx + s * dy) * 1e-3, (c * dy - s * dx) * 1e-3,
                           (c * dvx + s * dvy) * 1e-2, (c * dvy - s * dvx) * 1e-2, seen], -1)
        return (ego * seen[..., None]).flatten(-2)
