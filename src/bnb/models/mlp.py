"""The skeleton's baseline model: a small MLP over one team's view.

Each own ship reads its own row plus a summary of the whole view (every ship row, the
fields and the globals), raw apart from fixed scales. It exists to exercise the loop, not
to play well; Yemong is the model that matters. The first layer is split along that
concatenation, so the summary half is computed once per view instead of once per ship.
The model reads token position (the summary is a flattened view), so it relies on rows
being ordered by team_relative. It reads its rows by width, so it is built for a list of
columns per part (`observation`, the code's contract by default; a loaded policy file's
recorded ones) and selects those columns by name from the code's observation.
"""

import dataclasses

import torch
import torch.nn as nn

from bnb.contracts import Obs
from bnb.contracts.obs import COLUMNS, FIELD, GLOBAL, SHIP
from bnb.models.initialise import initialise

from .heads import SIZES

SHIP_SCALE = {
    "x": 1e-3,
    "y": 1e-3,
    "velocity_x": 1e-2,
    "velocity_y": 1e-2,
    "proper_speed": 1e-2,
    "energy": 1e-4,
    "shield": 1e-2,
    "respawn_wait": 0.125,
    "log_index_gradient_x": 100.0,
    "log_index_gradient_y": 100.0,
}
FIELD_SCALE = {"x": 1e-3, "y": 1e-3, "radius": 1e-2, "band_width": 1e-2}
GLOBAL_SCALE = {
    "arena_radius": 1e-3,
    "score_own": 0.04,
    "score_enemy": 0.04,
    "score_target": 0.04,
    "front": 1.0 / 3.0,
    "front_threshold": 1.0 / 3.0,
}


@dataclasses.dataclass(frozen=True)
class MLPBuild:
    """Everything the constructor takes (runs spec 4): the fixed sizes the summary is laid
    out for, and the hidden width."""

    ships_per_team: int
    num_ships: int
    num_fields: int
    hidden: int = 256


class MLP(nn.Module):
    """Built from an `MLPBuild` and the observation columns it reads, by part (the code's
    `COLUMNS` by default). VERSION rises whenever a change alters what saved weights compute
    (runs spec 4)."""

    NAME = "mlp"
    VERSION = 1
    reads = ("ships", "fields", "globals")
    reads_by_width = True  # built for recorded columns (`bnb.runs.checkpoint.build_model`)

    def __init__(self, build: MLPBuild, observation: dict | None = None, seed: int = 0) -> None:
        super().__init__()
        columns = dict(COLUMNS, **(observation or {}))
        self.observation = {part: tuple(columns[part]) for part in COLUMNS}
        ship_columns, field_columns, global_columns = (
            self.observation[part] for part in ("ships", "fields", "globals")
        )
        C, hidden = len(ship_columns), build.hidden
        self.build, self.ships_per_team = build, build.ships_per_team
        summary = build.num_ships * C + build.num_fields * len(field_columns) + len(global_columns)
        self.own = nn.Linear(C, hidden)
        self.summary = nn.Linear(summary, hidden, bias=False)
        self.body = nn.Linear(hidden, hidden)
        self.policy = nn.Linear(hidden, sum(SIZES))
        self.value = nn.Linear(hidden, 1)
        scale = torch.tensor([SHIP_SCALE.get(name, 1.0) for name in ship_columns])
        self.register_buffer("ship_scale", scale, persistent=False)
        scale = torch.tensor([FIELD_SCALE.get(name, 1.0) for name in field_columns])
        self.register_buffer("field_scale", scale, persistent=False)
        self.register_buffer(
            "global_scale",
            torch.tensor([GLOBAL_SCALE.get(name, 1.0) for name in global_columns]),
            persistent=False,
        )
        # The adapter: each part's columns by name from the code's observation (None when
        # they are the code's own, so the read is the observation as it is).
        self.selected = {
            part: None if names == COLUMNS[part] else [index[name] for name in names]
            for part, names, index in (
                ("ships", ship_columns, SHIP),
                ("fields", field_columns, FIELD),
                ("globals", global_columns, GLOBAL),
            )
        }
        self.since = ship_columns.index("time_since_seen")
        initialise(self, seed)

    def initialise_parameters(self, generator) -> None:
        """The policy head zero: a uniform policy at the start (no draw)."""
        nn.init.zeros_(self.policy.weight)
        nn.init.zeros_(self.policy.bias)

    def _part(self, obs: Obs, part: str) -> torch.Tensor:
        value, selected = getattr(obs, part), self.selected[part]
        return value if selected is None else value[..., selected]

    def forward(self, obs: Obs) -> tuple[torch.Tensor, torch.Tensor]:
        """Logits (..., S, sum(SIZES)) and values (..., S) for views with any leading axes."""
        ships = self._part(obs, "ships") * self.ship_scale
        since = self.since
        ships = torch.cat(
            [
                ships[..., :since],
                torch.log1p(ships[..., since : since + 1]),
                ships[..., since + 1 :],
            ],
            -1,
        )
        summary = torch.cat(
            [
                ships.flatten(-2),
                (self._part(obs, "fields") * self.field_scale).flatten(-2),
                self._part(obs, "globals") * self.global_scale,
            ],
            -1,
        )
        own = ships[..., : self.ships_per_team, :]
        h = torch.relu(self.own(own) + self.summary(summary)[..., None, :])
        h = torch.relu(self.body(h))
        return self.policy(h), self.value(h)[..., 0]
