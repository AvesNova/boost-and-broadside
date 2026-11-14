"""The nearest-neighbour MLP: a stateless per-own-ship MLP over a fixed-size input that does
not depend on the fleet or the field count, so a model trained at 5v5 plays any size, with
the relational model's outputs (the joint 42-way policy, the categorical critic and the
outcome) and no other heads.

Each own ship reads, in this order (`input_width`):

- its own state: the relational model's 15 own features (`relational.features`), then the
  mode's globals (`GLOBALS`: episode progress, the arena radius, fog, the mode one-hot, the
  scores own, enemy and target, the front and its threshold; scaled as the MLP's
  `GLOBAL_SCALE`), 25 in all;
- its `allies` nearest living allies and its `enemies` nearest living visible enemies, by
  distance, nearest first (ties by row), each with the relational model's 10 features of
  that ship relative to it and a presence flag (11); a slot without a candidate is zero.
  A hidden enemy is never a candidate: the model has no memory and sees what is visible;
- its `fields` nearest fields by the distance to their centre, nearest first, each with
  the offset of its centre in the own heading frame, the distance to the centre, the
  signed distance to its edge (centre distance minus radius), its radius, band width and
  level, and a presence flag (8);
- every zone slot by role (own spawn, own defense, neutral, enemy defense, enemy spawn;
  `ZONE_ROLES`), so a slot always means the same zone: the offset of its centre in the own
  heading frame, the distance, its radius, the role one-hot, its progress and direction,
  the symlog of the captures to win and to lose, and a presence flag (14); a role no
  present zone has (a mode without zones) is zero.

Two hidden GELU layers per own ship feed the joint policy and the critic; the outcome reads
the mean of the own ships' hidden states, as the relational model's. The own ships are the
first half of the rows (`TEAMS` teams of equal size), read from the view's shape, so the
build record holds no fleet size.
"""

import dataclasses

import torch
import torch.nn as nn

from bnb.contracts import REWARD_LEVELS, Obs
from bnb.contracts.action import NUM_JOINT
from bnb.contracts.config import TEAMS
from bnb.contracts.obs import FIELD, GLOBAL, SHIP, ZONE, ZONE_SLOTS
from bnb.models.initialise import initialise

from . import relational
from .mlp import GLOBAL_SCALE
from .relational import DISTANCE_SCALE, OTHER_FEATURES, OWN_FEATURES, Unrolled, _rotate
from .yemong import critic
from .yemong.heads import critic_values

# The globals each own ship reads, with their scales (the MLP's where it has one).
GLOBALS = (
    ("episode_progress", 1.0),
    ("arena_radius", GLOBAL_SCALE["arena_radius"]),
    ("fogged", 1.0),
    ("mode_team_deathmatch", 1.0),
    ("mode_frontline", 1.0),
    ("score_own", GLOBAL_SCALE["score_own"]),
    ("score_enemy", GLOBAL_SCALE["score_enemy"]),
    ("score_target", GLOBAL_SCALE["score_target"]),
    ("front", GLOBAL_SCALE["front"]),
    ("front_threshold", GLOBAL_SCALE["front_threshold"]),
)
# Yemong's field and zone scales (`yemong.inputs`), constants of this model too.
FIELD_RADIUS_SCALE = 1000.0  # m
FIELD_BAND_SCALE = 40.0  # m
FIELD_LEVEL_SCALE = 2.0
ZONE_ROLES = 5  # own spawn, own defense, neutral, enemy defense, enemy spawn
SELF_FEATURES = OWN_FEATURES + len(GLOBALS)
SHIP_FEATURES = OTHER_FEATURES + 1  # the relational features and presence
FIELD_FEATURES = 8
ZONE_FEATURES = 3 + 1 + ZONE_ROLES + 4 + 1
# The columns of the relational model's pair features (`relational.features`).
PAIR_DISTANCE, PAIR_TEAM, PAIR_ALIVE, PAIR_VISIBLE = 4, 7, 8, 9


@dataclasses.dataclass(frozen=True)
class NearestConfig:
    """hidden: the width of both hidden layers; allies, enemies, fields: how many of the
    nearest of each every own ship reads; levels and outcome_level as Yemong's."""

    hidden: int = 256
    allies: int = 3
    enemies: int = 4
    fields: int = 3
    value_bins: int = 77  # symlog bins 0.079 apart, one at 0 (as 51 over +-2 were)
    value_limit: float = 3.0  # symlog: +-19 in reward units
    levels: tuple[str, ...] = REWARD_LEVELS
    outcome_level: str = "outcome"
    bfloat16_matmuls: bool = False

    @property
    def local_levels(self) -> tuple[int, ...]:
        """The indices into levels of every level but outcome_level: the levels with a value
        distribution."""
        return tuple(i for i, name in enumerate(self.levels) if name != self.outcome_level)


@dataclasses.dataclass(frozen=True)
class NearestBuild:
    """Everything the constructor takes (runs spec 4): the config alone; no fleet or field
    count enters the input."""

    config: NearestConfig


def input_width(config: NearestConfig) -> int:
    """The width of each own ship's input: the same at every fleet size and field count."""
    return (
        SELF_FEATURES
        + (config.allies + config.enemies) * SHIP_FEATURES
        + config.fields * FIELD_FEATURES
        + ZONE_SLOTS * ZONE_FEATURES
    )


def _nearest(distance: torch.Tensor, candidate: torch.Tensor, count: int):
    """The indices (..., count) of the `count` nearest candidates along the last axis (ties
    by position) and their presence (..., count), bool; a slot past the candidates points at
    a padding column (index >= the axis's size) with presence False."""
    pad = max(0, count - distance.shape[-1])
    if pad:
        distance = torch.cat([distance, distance.new_zeros(distance.shape[:-1] + (pad,))], -1)
        candidate = torch.cat([candidate, candidate.new_zeros(candidate.shape[:-1] + (pad,))], -1)
    key = torch.where(candidate, distance, torch.inf)
    order = torch.sort(key, dim=-1, stable=True).indices[..., :count]
    return order, candidate.gather(-1, order)


def _pick(rows: torch.Tensor, order: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    """(..., count, F + 1): the rows (..., n, F) at `order` (..., count), padded with zero
    rows for an index past n, each with its presence; zero where absent."""
    pad = max(0, int(order.shape[-1]) - rows.shape[-2])
    if pad:
        rows = torch.cat([rows, rows.new_zeros(rows.shape[:-2] + (pad, rows.shape[-1]))], -2)
    picked = rows.gather(-2, order[..., None].expand(order.shape + (rows.shape[-1],)))
    picked = torch.where(present[..., None], picked, 0.0)
    return torch.cat([picked, present[..., None].to(picked.dtype)], -1)


def _relative(
    x: torch.Tensor, y: torch.Tensor, own_x, own_y, cos, sin
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Points (..., P) relative to each own ship (..., S, 1): forward, left and distance
    (..., S, P), over DISTANCE_SCALE."""
    forward, left = _rotate(x[..., None, :] - own_x, y[..., None, :] - own_y, cos, sin)
    distance = torch.sqrt(forward * forward + left * left)
    return forward / DISTANCE_SCALE, left / DISTANCE_SCALE, distance / DISTANCE_SCALE


def features(obs: Obs, config: NearestConfig) -> torch.Tensor:
    """(..., S, input_width(config)): each own ship's inputs (see the module docstring)."""
    ships = obs.ships
    rows = ships.shape[-2]
    own = rows // TEAMS
    paired = relational.features(obs, own)
    own_state = paired[..., :OWN_FEATURES]
    # Row j' of the others is row j' + (j' >= i) of the view for own ship i.
    others = paired[..., OWN_FEATURES:].unflatten(-1, (rows - 1, OTHER_FEATURES))  # (.., S, N-1, F)
    alive, visible = others[..., PAIR_ALIVE] > 0.5, others[..., PAIR_VISIBLE] > 0.5
    ally = others[..., PAIR_TEAM] == 0.0
    distance = others[..., PAIR_DISTANCE]
    allies = _pick(others, *_nearest(distance, ally & alive & visible, config.allies))
    enemies = _pick(others, *_nearest(distance, ~ally & alive & visible, config.enemies))

    globals_ = obs.globals
    scalars = torch.stack([globals_[..., GLOBAL[name]] * scale for name, scale in GLOBALS], -1)
    scalars = scalars[..., None, :].expand(own_state.shape[:-1] + (len(GLOBALS),))

    me = slice(None, own)
    own_x, own_y, cos, sin = (
        ships[..., me, SHIP[name]][..., None] for name in ("x", "y", "heading_cos", "heading_sin")
    )  # each (..., S, 1)

    fields = obs.fields
    forward, left, centre = _relative(
        fields[..., FIELD["x"]], fields[..., FIELD["y"]], own_x, own_y, cos, sin
    )
    radius = fields[..., FIELD["radius"]][..., None, :].expand(centre.shape)
    field_rows = torch.stack(
        [
            forward,
            left,
            centre,
            centre - radius / DISTANCE_SCALE,
            radius / FIELD_RADIUS_SCALE,
            fields[..., FIELD["band_width"]][..., None, :].expand(centre.shape) / FIELD_BAND_SCALE,
            fields[..., FIELD["level"]][..., None, :].expand(centre.shape) / FIELD_LEVEL_SCALE,
        ],
        -1,
    )  # (..., S, M, 7)
    every = torch.ones_like(centre, dtype=torch.bool)
    near_fields = _pick(field_rows, *_nearest(centre, every, config.fields))

    zones = by_role(obs.zones)  # (..., ZONE_ROLES, len(ZONE_COLUMNS))
    present = zones[..., ZONE["present"]] > 0.5
    forward, left, distance = _relative(
        zones[..., ZONE["x"]], zones[..., ZONE["y"]], own_x, own_y, cos, sin
    )
    shape = distance.shape  # (..., S, ZONE_ROLES)

    def per_zone(column: str) -> torch.Tensor:
        return zones[..., ZONE[column]][..., None, :].expand(shape)

    zone_rows = torch.cat(
        [
            torch.stack([forward, left, distance, per_zone("radius") / FIELD_RADIUS_SCALE], -1),
            torch.eye(ZONE_ROLES, device=ships.device).expand(shape + (ZONE_ROLES,)),
            torch.stack(
                [
                    per_zone("progress"),
                    per_zone("direction"),
                    critic.symlog(per_zone("offensive_distance")),
                    critic.symlog(per_zone("defensive_distance")),
                ],
                -1,
            ),
        ],
        -1,
    )
    present = present[..., None, :].expand(shape)
    zone_rows = torch.where(present[..., None], zone_rows, 0.0)
    zone_rows = torch.cat([zone_rows, present[..., None].to(zone_rows.dtype)], -1)
    return torch.cat(
        [
            own_state,
            scalars,
            allies.flatten(-2),
            enemies.flatten(-2),
            near_fields.flatten(-2),
            zone_rows.flatten(-2),
        ],
        -1,
    )


def by_role(zones: torch.Tensor) -> torch.Tensor:
    """(..., ZONE_ROLES, len(ZONE_COLUMNS)): the zone rows (..., Z, ...) in role order, each
    role's present zone (the first, should two share one); a role no present zone has is a
    zero row (its `present` 0). A gather of static shape: no host sync."""
    role = zones[..., ZONE["role"]].round().long()
    present = zones[..., ZONE["present"]] > 0.5
    match = (role[..., None, :] == torch.arange(ZONE_ROLES, device=zones.device)[:, None]) & (
        present[..., None, :]
    )  # (..., ZONE_ROLES, Z)
    index = match.float().argmax(-1)  # the first match; 0 when none
    rows = zones.gather(-2, index[..., None].expand(index.shape + (zones.shape[-1],)))
    return torch.where(match.any(-1)[..., None], rows, 0.0)


class Nearest(nn.Module):
    """Built from a `NearestBuild`. VERSION rises whenever a change alters what saved weights
    compute (runs spec 4)."""

    NAME = "nearest"
    VERSION = 1
    reads = ("ships", "fields", "globals", "zones")

    def __init__(self, build: NearestBuild, seed: int = 0) -> None:
        super().__init__()
        config = build.config
        self.build, self.config = build, config
        hidden = config.hidden
        self.body = nn.Sequential(
            nn.Linear(input_width(config), hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        levels = len(config.levels) - 1
        self.policy = nn.Linear(hidden, NUM_JOINT)
        self.value = nn.Linear(hidden, levels * config.value_bins)
        self.outcome = nn.Linear(hidden, critic.OUTCOMES)
        self.register_buffer(
            "bins", critic.value_bins(config.value_bins, config.value_limit), persistent=False
        )
        initialise(self, seed)

    def initialise_parameters(self, generator) -> None:
        """The heads zero: a uniform policy and level values at the start (no draw)."""
        for head in (self.policy, self.value, self.outcome):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def initial_carry(self, views: int, device=None) -> None:
        """No carry: the model is stateless."""
        return None

    def forward(self, obs: Obs) -> Unrolled:
        """Views with any leading axes, at any fleet size and field count."""
        config = self.config
        inputs = features(obs, config)
        with torch.autocast(obs.ships.device.type, torch.bfloat16, enabled=config.bfloat16_matmuls):
            hidden = self.body(inputs)
            policy_logits = self.policy(hidden)
            value_logits = self.value(hidden)
            outcome_logits = self.outcome(hidden.mean(-2))
        value_logits = value_logits.float().unflatten(-1, (-1, config.value_bins))
        outcome_logits = outcome_logits.float()
        values = critic_values(
            value_logits,
            outcome_logits,
            self.bins,
            config.local_levels,
            config.levels.index(config.outcome_level),
        )
        return Unrolled(
            policy_logits=policy_logits.float(),
            value_logits=value_logits,
            values=values,
            outcome_logits=outcome_logits,
        )

    def unroll(self, carry: None, obs: Obs, replay: None, reset: torch.Tensor) -> Unrolled:
        """Stateless: a stored segment is the per-step forward over every step at once."""
        return self(obs)
