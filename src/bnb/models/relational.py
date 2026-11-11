"""The toy learner's model (learner spec 9): a fixed-size relational MLP for small team
deathmatch, stateless, with the joint 42-way policy and the categorical critic.

Each own ship reads its own state and, for every other row of the view (allies first, then
enemies, in row order), that ship relative to it: position and velocity rotated into its
heading frame, distance, relative heading (cos, sin), team, alive and visible. A hidden
ship's relative columns are zero (its observed state is). Two hidden layers per own ship;
the outcome reads the mean of the own ships' hidden states.
"""

import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F

from bnb.contracts import REWARD_LEVELS, Obs
from bnb.contracts.action import NUM_JOINT
from bnb.contracts.obs import GLOBAL, SHIP
from bnb.contracts.tree import contract

from .yemong import critic
from .yemong.heads import critic_values

# Each input's typical size, so the features are of order one: a fixed scale, not a game
# rule, and nothing breaks if a value exceeds it.
DISTANCE_SCALE = 1000.0  # m: the pair offsets and distances
SPEED_SCALE = 100.0  # m/s: the velocities
ENERGY_SCALE = 1e4  # J: the energy store
SHIELD_SCALE = 100.0  # points: the shield
SHIELD_WAIT_SCALE = 6.0  # s: the shield's recharge wait
GUN_COOLDOWN_RATE = 10.0  # 1/s: the gun's cooldown, in tenths of a second
RESPAWN_WAIT_SCALE = 5.0  # s: the respawn wait
OWN_FEATURES = 15
OTHER_FEATURES = 10


@dataclasses.dataclass(frozen=True)
class RelationalConfig:
    """hidden: the width of both hidden layers; levels and outcome_level as Yemong's."""

    hidden: int = 256
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


@contract
class Unrolled:
    """A segment's outputs, (L, R, ...) leading: policy_logits (S, 42), value_logits
    (S, levels - 1, bins), values (S, levels), outcome_logits (4). The model has no other heads."""

    policy_logits: torch.Tensor
    value_logits: torch.Tensor
    values: torch.Tensor
    outcome_logits: torch.Tensor


def _rotate(x: torch.Tensor, y: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """(x, y) in the frame of a heading (cos, sin): forward, left."""
    return x * cos + y * sin, y * cos - x * sin


def features(obs: Obs, own: int) -> torch.Tensor:
    """(..., S, OWN_FEATURES + (N - 1) OTHER_FEATURES): each own ship's inputs."""
    ships = obs.ships
    column = {name: ships[..., SHIP[name]] for name in SHIP}
    rows = ships.shape[-2]
    radius = obs.globals[..., GLOBAL["arena_radius"]][..., None].clamp_min(1.0)
    cos, sin = column["heading_cos"], column["heading_sin"]
    me = slice(None, own)
    x, y = column["x"][..., me], column["y"][..., me]
    centre_forward, centre_left = _rotate(-x, -y, cos[..., me], sin[..., me])
    velocity_forward, velocity_left = _rotate(
        column["velocity_x"][..., me], column["velocity_y"][..., me], cos[..., me], sin[..., me]
    )
    own_state = torch.stack(
        [
            centre_forward / radius,
            centre_left / radius,
            torch.sqrt(x * x + y * y) / radius,
            velocity_forward / SPEED_SCALE,
            velocity_left / SPEED_SCALE,
            column["angle_of_attack"][..., me],
            column["nose_rate"][..., me],
            column["energy"][..., me] / ENERGY_SCALE,
            column["shield"][..., me] / SHIELD_SCALE,
            column["shield_recharge_wait"][..., me] / SHIELD_WAIT_SCALE,
            column["gun_cooldown"][..., me] * GUN_COOLDOWN_RATE,
            column["alive"][..., me],
            column["respawn_wait"][..., me] / RESPAWN_WAIT_SCALE,
            column["ship_type"][..., me],
            column["log_refractive_index"][..., me],
        ],
        -1,
    )

    # Pairs (own i, row j): j relative to i, in i's heading frame.
    def pair(values: torch.Tensor) -> torch.Tensor:
        return values[..., None, :] - values[..., me, None]

    offset_forward, offset_left = _rotate(
        pair(column["x"]), pair(column["y"]), cos[..., me, None], sin[..., me, None]
    )
    moving_forward, moving_left = _rotate(
        pair(column["velocity_x"]),
        pair(column["velocity_y"]),
        cos[..., me, None],
        sin[..., me, None],
    )
    heading_cos = cos[..., me, None] * cos[..., None, :] + sin[..., me, None] * sin[..., None, :]
    heading_sin = cos[..., me, None] * sin[..., None, :] - sin[..., me, None] * cos[..., None, :]
    visible = column["visible"][..., None, :].expand(offset_forward.shape)
    distance = torch.sqrt(offset_forward**2 + offset_left**2)
    seen = torch.stack(
        [
            offset_forward / DISTANCE_SCALE,
            offset_left / DISTANCE_SCALE,
            moving_forward / SPEED_SCALE,
            moving_left / SPEED_SCALE,
            distance / DISTANCE_SCALE,
            heading_cos,
            heading_sin,
        ],
        -1,
    )
    seen = torch.where(visible[..., None] > 0.5, seen, 0.0)
    team = column["team_relative"][..., None, :].expand(visible.shape)
    alive = column["alive"][..., None, :].expand(visible.shape)
    others = torch.cat([seen, torch.stack([team, alive, visible], -1)], -1)  # (..., S, N, F)
    # Drop each own ship's own row: own ship i reads rows 0 .. i - 1, then i + 1 .. N - 1,
    # by a gather of static indices (no boolean index, so no host sync).
    other = torch.arange(rows - 1, device=ships.device)
    index = other + (other >= torch.arange(own, device=ships.device)[:, None])  # (S, N - 1)
    index = index[..., None].expand(others.shape[:-2] + (rows - 1, others.shape[-1]))
    others = others.gather(-2, index)
    return torch.cat([own_state, others.flatten(-2)], -1)


@dataclasses.dataclass(frozen=True)
class RelationalBuild:
    """Everything the constructor takes (runs spec 4): the config and the fixed fleet sizes
    the inputs are laid out for."""

    config: RelationalConfig
    ships_per_team: int
    num_ships: int


class Relational(nn.Module):
    """Built from a `RelationalBuild`. VERSION rises whenever a change alters what saved
    weights compute (runs spec 4)."""

    NAME = "relational"
    VERSION = 1
    reads = ("ships", "globals")

    def __init__(self, build: RelationalBuild) -> None:
        super().__init__()
        config = build.config
        self.build, self.config, self.ships_per_team = build, config, build.ships_per_team
        inputs = OWN_FEATURES + (build.num_ships - 1) * OTHER_FEATURES
        hidden = config.hidden
        self.body = nn.Sequential(
            nn.Linear(inputs, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU()
        )
        levels = len(config.levels) - 1
        self.policy = nn.Linear(hidden, NUM_JOINT)
        self.value = nn.Linear(hidden, levels * config.value_bins)
        self.outcome = nn.Linear(hidden, critic.OUTCOMES)
        for head in (self.policy, self.value, self.outcome):  # uniform at the start
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.register_buffer(
            "bins", critic.value_bins(config.value_bins, config.value_limit), persistent=False
        )

    def initial_carry(self, views: int, device=None) -> None:
        """No carry: the model is stateless."""
        return None

    def forward(self, obs: Obs) -> Unrolled:
        """Views with any leading axes."""
        config = self.config
        with torch.autocast(obs.ships.device.type, torch.bfloat16, enabled=config.bfloat16_matmuls):
            hidden = self.body(features(obs, self.ships_per_team))
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


def outcome_probabilities(unrolled: Unrolled) -> torch.Tensor:
    """(..., OUTCOMES): the outcome probabilities (win, tie, loss, unresolved), the softmax of the
    outcome logits."""
    return F.softmax(unrolled.outcome_logits, -1)
