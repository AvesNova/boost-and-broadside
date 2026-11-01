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

DISTANCE_SCALE = 1000.0  # m
SPEED_SCALE = 100.0  # m/s
OWN_FEATURES = 15
OTHER_FEATURES = 10


@dataclasses.dataclass(frozen=True)
class RelationalConfig:
    """hidden: the width of both hidden layers; levels and outcome_level as Yemong's."""

    hidden: int = 256
    value_bins: int = 51
    value_limit: float = 2.0
    levels: tuple[str, ...] = REWARD_LEVELS
    outcome_level: str = "outcome"
    bfloat16_matmuls: bool = False

    @property
    def local_levels(self) -> tuple[int, ...]:
        return tuple(i for i, name in enumerate(self.levels) if name != self.outcome_level)


@contract
class Unrolled:
    """A segment's outputs, (L, R, ...) leading: policy_logits (S, 42), value_logits
    (S, K - 1, bins), values (S, K), outcome_logits (4). The model has no other heads."""

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
    velocity_forward, velocity_left = _rotate(column["velocity_x"][..., me],
                                              column["velocity_y"][..., me], cos[..., me],
                                              sin[..., me])
    own_state = torch.stack([
        centre_forward / radius, centre_left / radius,
        torch.sqrt(x * x + y * y) / radius,
        velocity_forward / SPEED_SCALE, velocity_left / SPEED_SCALE,
        column["angle_of_attack"][..., me], column["nose_rate"][..., me],
        column["energy"][..., me] / 1e4, column["shield"][..., me] / 100.0,
        column["shield_recharge_wait"][..., me] / 6.0, column["gun_cooldown"][..., me] * 10.0,
        column["alive"][..., me], column["respawn_wait"][..., me] / 5.0,
        column["ship_type"][..., me], column["log_refractive_index"][..., me]], -1)

    # Pairs (own i, row j): j relative to i, in i's heading frame.
    def pair(values: torch.Tensor) -> torch.Tensor:
        return values[..., None, :] - values[..., me, None]

    offset_forward, offset_left = _rotate(pair(column["x"]), pair(column["y"]),
                                          cos[..., me, None], sin[..., me, None])
    moving_forward, moving_left = _rotate(pair(column["velocity_x"]), pair(column["velocity_y"]),
                                          cos[..., me, None], sin[..., me, None])
    heading_cos = cos[..., me, None] * cos[..., None, :] + sin[..., me, None] * sin[..., None, :]
    heading_sin = cos[..., me, None] * sin[..., None, :] - sin[..., me, None] * cos[..., None, :]
    visible = column["visible"][..., None, :].expand(offset_forward.shape)
    distance = torch.sqrt(offset_forward**2 + offset_left**2)
    seen = torch.stack([offset_forward / DISTANCE_SCALE, offset_left / DISTANCE_SCALE,
                        moving_forward / SPEED_SCALE, moving_left / SPEED_SCALE,
                        distance / DISTANCE_SCALE, heading_cos, heading_sin], -1)
    seen = torch.where(visible[..., None] > 0.5, seen, 0.0)
    team = column["team_relative"][..., None, :].expand(visible.shape)
    alive = column["alive"][..., None, :].expand(visible.shape)
    others = torch.cat([seen, torch.stack([team, alive, visible], -1)], -1)  # (..., S, N, F)
    # Drop each own ship's own row: the others stay in row order.
    keep = ~torch.eye(own, rows, dtype=torch.bool, device=ships.device)
    others = others[..., keep, :].unflatten(-2, (own, rows - 1))
    return torch.cat([own_state, others.flatten(-2)], -1)


class Relational(nn.Module):
    reads = ("ships", "globals")

    def __init__(self, config: RelationalConfig, ships_per_team: int, num_ships: int) -> None:
        super().__init__()
        self.config, self.ships_per_team = config, ships_per_team
        inputs = OWN_FEATURES + (num_ships - 1) * OTHER_FEATURES
        hidden = config.hidden
        self.body = nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(),
                                  nn.Linear(hidden, hidden), nn.GELU())
        levels = len(config.levels) - 1
        self.policy = nn.Linear(hidden, NUM_JOINT)
        self.value = nn.Linear(hidden, levels * config.value_bins)
        self.outcome = nn.Linear(hidden, critic.OUTCOMES)
        for head in (self.policy, self.value, self.outcome):  # uniform at the start
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.register_buffer("bins", critic.value_bins(config.value_bins, config.value_limit),
                             persistent=False)

    def initial_carry(self, views: int, device=None) -> None:
        return None

    def forward(self, obs: Obs) -> Unrolled:
        """Views with any leading axes."""
        config = self.config
        with torch.autocast(obs.ships.device.type, torch.bfloat16,
                            enabled=config.bfloat16_matmuls):
            hidden = self.body(features(obs, self.ships_per_team))
            policy_logits = self.policy(hidden)
            value_logits = self.value(hidden)
            outcome_logits = self.outcome(hidden.mean(-2))
        value_logits = value_logits.float().unflatten(-1, (-1, config.value_bins))
        outcome_logits = outcome_logits.float()
        values = critic_values(value_logits, outcome_logits, self.bins, config.local_levels,
                               config.levels.index(config.outcome_level))
        return Unrolled(policy_logits=policy_logits.float(), value_logits=value_logits,
                        values=values, outcome_logits=outcome_logits)

    def unroll(self, carry: None, obs: Obs, replay: None, reset: torch.Tensor) -> Unrolled:
        """Stateless: a stored segment is the per-step forward over every step at once."""
        return self(obs)


def outcome_probabilities(unrolled: Unrolled) -> torch.Tensor:
    return F.softmax(unrolled.outcome_logits, -1)
