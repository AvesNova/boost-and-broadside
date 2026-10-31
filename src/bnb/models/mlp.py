"""The skeleton's baseline model: a small MLP over one team's view.

Each own ship reads its own row plus a summary of the whole view (every ship row, the
fields and the globals), raw apart from fixed scales. It exists to exercise the loop, not
to play well; Yemong is the model that matters. The first layer is split along that
concatenation, so the summary half is computed once per view instead of once per ship.
The model reads token position (the summary is a flattened view), so it relies on rows
being ordered by team_rel.
"""

import torch
import torch.nn as nn

from bnb.contracts import Obs
from bnb.contracts.obs import FIELD_COLUMNS, GLOBAL_COLUMNS, SHIP, SHIP_COLUMNS

from .heads import SIZES

SHIP_SCALE = {"x": 1e-3, "y": 1e-3, "vx": 1e-2, "vy": 1e-2, "w": 1e-2, "E": 1e-4,
              "shield": 1e-2, "respawn": 0.125}
FIELD_SCALE = (1e-3, 1e-3, 1e-2, 1.0)
GLOBAL_SCALE = (1.0, 1e-3, 1.0, 1.0)


class MLP(nn.Module):
    reads = ("ships", "fields", "globals")

    def __init__(self, ships_per_team: int, num_ships: int, num_fields: int, hidden: int = 256):
        super().__init__()
        C = len(SHIP_COLUMNS)
        self.ships_per_team = ships_per_team
        summary = num_ships * C + num_fields * len(FIELD_COLUMNS) + len(GLOBAL_COLUMNS)
        self.own = nn.Linear(C, hidden)
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
        h = torch.relu(self.own(own) + self.summary(summary)[..., None, :])
        h = torch.relu(self.body(h))
        return self.policy(h), self.value(h)[..., 0]

