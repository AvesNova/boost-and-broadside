"""Unified token encoder for ships and obstacles.

A single UnifiedEncoder handles both entity types by building a shared raw
feature vector of the same dimension. Features that don't apply to an entity
type are zero-padded, so the feature_extractor MLP is shared with no separate
weights.

Feature layout (per token):
    pos_feat          4n   shared     — Fourier (x, y)
    att_feat          4n   ship-only  — Fourier (nose att, vel att); zeros for obstacles
    vel_feat          2    shared     — symlog (vx, vy)
    ang_vel           1    ship-only  — zeros for obstacles
    scalars           3    ship-only  — [health, power, cooldown]; zeros for obstacles
    team_onehot       3    shared     — one_hot(team_id, 3): team0 / team1 / obstacle
    collision_radius  1    shared     — normalized; ship: constant; obstacle: obs_radius
    alive             1    ship-only  — zeros for obstacles
    action_feat       12   ship-only  — prev-action one-hot; zeros for obstacles
    gc_feat           4n   obstacle-only — Fourier (gravity center x, y); zeros for ships
    obs_hit           1    obstacle-only — collision flag; zeros for ships
    ─────────────────────────────────────────────────────────────────────
    Total             12n + 24                               (120 at n=8)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from boost_and_broadside.config import ModelConfig, ShipConfig
from boost_and_broadside.constants import (
    NUM_POWER_ACTIONS,
    NUM_TURN_ACTIONS,
    NUM_SHOOT_ACTIONS,
)


_TEAM_CLASSES = 3   # team0, team1, obstacle
_ACTION_DIM = NUM_POWER_ACTIONS + NUM_TURN_ACTIONS + NUM_SHOOT_ACTIONS  # 12


def _raw_dim(n_fourier_freqs: int) -> int:
    n = n_fourier_freqs
    return (
        4 * n      # pos_feat
        + 4 * n    # att_feat (ship-only)
        + 2        # vel_feat
        + 1        # ang_vel (ship-only)
        + 3        # scalars (ship-only)
        + _TEAM_CLASSES  # team_onehot
        + 1        # collision_radius
        + 1        # alive (ship-only)
        + _ACTION_DIM  # action_feat (ship-only)
        + 4 * n    # gc_feat (obstacle-only)
        + 1        # obs_hit (obstacle-only)
    )  # = 12n + 24


def _symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def _fourier_encode(
    coords: torch.Tensor, n_freqs: int, max_period: float
) -> torch.Tensor:
    """Fourier-encode scalar coordinates using base-2 power frequencies."""
    k = torch.arange(n_freqs, device=coords.device, dtype=torch.float32)
    freqs = (2.0 * torch.pi / max_period) * (2.0**k)
    args = coords.unsqueeze(-1) * freqs
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


def _fourier_encode_angle(cossin: torch.Tensor, n_freqs: int) -> torch.Tensor:
    """Fourier-encode a unit direction vector preserving 2π circularity."""
    angle = torch.atan2(cossin[..., 1], cossin[..., 0])
    k = torch.arange(n_freqs, device=angle.device, dtype=torch.float32)
    freqs = 2.0**k
    args = angle.unsqueeze(-1) * freqs
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class UnifiedEncoder(nn.Module):
    """Single encoder for both ships and obstacles.

    Ships and obstacles share the same feature_extractor MLP.
    Entity-specific features are zero-padded for the other type.
    """

    def __init__(self, model_config: ModelConfig, ship_config: ShipConfig) -> None:
        super().__init__()
        self.n_freqs = model_config.n_fourier_freqs
        self.d_model = model_config.d_model
        max_dim = float(max(ship_config.world_size))
        self.ship_cr_norm = ship_config.collision_radius / max_dim

        in_dim = _raw_dim(self.n_freqs)
        d = model_config.d_model
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_dim, 2 * d),
            nn.RMSNorm(2 * d),
            nn.GELU(),
            nn.Linear(2 * d, d),
            nn.RMSNorm(d),
        )

    def encode_ships(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode ship observations into tokens.

        Args:
            obs: Dict with ship keys from MVPEnvWrapper._get_obs():
                "pos"         (..., N, 2) float32 — normalized [0,1]
                "vel"         (..., N, 2) float32
                "att"         (..., N, 2) float32
                "ang_vel"     (..., N, 1) float32
                "scalars"     (..., N, 3) float32
                "team_id"     (..., N)   int32
                "alive"       (..., N)   bool
                "prev_action" (..., N, 3) int64

        Returns:
            (..., N, d_model) float32.
        """
        pos = obs["pos"]                        # (..., N, 2)
        vel = obs["vel"]                        # (..., N, 2)
        att = obs["att"]                        # (..., N, 2)
        ang_vel = obs["ang_vel"]                # (..., N, 1)
        scalars = obs["scalars"]                # (..., N, 3)
        team_id = obs["team_id"].long()         # (..., N)
        alive = obs["alive"]                    # (..., N)
        prev_action = obs["prev_action"].long() # (..., N, 3)

        # Position Fourier
        px_enc = _fourier_encode(pos[..., 0], self.n_freqs, 1.0)
        py_enc = _fourier_encode(pos[..., 1], self.n_freqs, 1.0)
        pos_feat = torch.cat([px_enc, py_enc], dim=-1)  # (..., N, 4n)

        # Attitude Fourier (ship-only)
        speed = torch.clamp(torch.norm(vel, dim=-1, keepdim=True), min=1e-6)
        vel_dir = vel / speed
        nose_enc = _fourier_encode_angle(att, self.n_freqs)
        vel_enc = _fourier_encode_angle(vel_dir, self.n_freqs)
        att_feat = torch.cat([nose_enc, vel_enc], dim=-1)  # (..., N, 4n)

        vel_feat = _symlog(vel)               # (..., N, 2)
        ang_feat = _symlog(ang_vel)           # (..., N, 1)

        team_feat = F.one_hot(team_id, _TEAM_CLASSES).float()  # (..., N, 3)

        cr_feat = pos.new_full((*pos.shape[:-1], 1), self.ship_cr_norm)  # (..., N, 1)

        alive_feat = alive.float().unsqueeze(-1)  # (..., N, 1)

        pa_power = F.one_hot(prev_action[..., 0], NUM_POWER_ACTIONS).float()
        pa_turn = F.one_hot(prev_action[..., 1], NUM_TURN_ACTIONS).float()
        pa_shoot = F.one_hot(prev_action[..., 2], NUM_SHOOT_ACTIONS).float()
        action_feat = torch.cat([pa_power, pa_turn, pa_shoot], dim=-1)  # (..., N, 12)

        # Obstacle-only slots → zeros for ships
        gc_feat = torch.zeros(*pos.shape[:-1], 4 * self.n_freqs, device=pos.device)
        obs_hit = torch.zeros(*pos.shape[:-1], 1, device=pos.device)

        raw = torch.cat([
            pos_feat,    # 4n  shared
            att_feat,    # 4n  ship-only
            vel_feat,    # 2   shared
            ang_feat,    # 1   ship-only
            scalars,     # 3   ship-only
            team_feat,   # 3   shared
            cr_feat,     # 1   shared
            alive_feat,  # 1   ship-only
            action_feat, # 12  ship-only
            gc_feat,     # 4n  obstacle-only (zero)
            obs_hit,     # 1   obstacle-only (zero)
        ], dim=-1)  # (..., N, 12n+24)

        return self.feature_extractor(raw)

    def encode_obstacles(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode obstacle observations into tokens.

        Args:
            obs: Dict with obstacle keys from MVPEnvWrapper._get_obs():
                "obstacle_pos":            (..., M, 2) float32 — normalized [0,1]
                "obstacle_vel":            (..., M, 2) float32
                "obstacle_radius":         (..., M, 1) float32 — normalized radius
                "obstacle_gravity_center": (..., M, 2) float32 — normalized [0,1]
                "obstacle_hit":            (..., M, 1) float32

        Returns:
            (..., M, d_model) float32.
        """
        obs_pos = obs["obstacle_pos"]               # (..., M, 2)
        obs_vel = obs["obstacle_vel"]               # (..., M, 2)
        obs_radius = obs["obstacle_radius"]         # (..., M, 1)
        obs_gcenter = obs["obstacle_gravity_center"]  # (..., M, 2)
        obs_hit = obs["obstacle_hit"]               # (..., M, 1)

        # Position Fourier
        px_enc = _fourier_encode(obs_pos[..., 0], self.n_freqs, 1.0)
        py_enc = _fourier_encode(obs_pos[..., 1], self.n_freqs, 1.0)
        pos_feat = torch.cat([px_enc, py_enc], dim=-1)  # (..., M, 4n)

        # Ship-only slots → zeros for obstacles
        att_feat = torch.zeros(*obs_pos.shape[:-1], 4 * self.n_freqs, device=obs_pos.device)
        ang_feat = torch.zeros(*obs_pos.shape[:-1], 1, device=obs_pos.device)
        scalars = torch.zeros(*obs_pos.shape[:-1], 3, device=obs_pos.device)
        alive_feat = torch.zeros(*obs_pos.shape[:-1], 1, device=obs_pos.device)
        action_feat = torch.zeros(*obs_pos.shape[:-1], _ACTION_DIM, device=obs_pos.device)

        vel_feat = _symlog(obs_vel)  # (..., M, 2)

        # Team one-hot: class 2 = obstacle
        team_feat = obs_pos.new_zeros(*obs_pos.shape[:-1], _TEAM_CLASSES)
        team_feat[..., 2] = 1.0  # obstacle class

        cr_feat = obs_radius  # already normalized same way as ship collision_radius

        # Gravity-center Fourier (obstacle-only)
        gx_enc = _fourier_encode(obs_gcenter[..., 0], self.n_freqs, 1.0)
        gy_enc = _fourier_encode(obs_gcenter[..., 1], self.n_freqs, 1.0)
        gc_feat = torch.cat([gx_enc, gy_enc], dim=-1)  # (..., M, 4n)

        raw = torch.cat([
            pos_feat,    # 4n  shared
            att_feat,    # 4n  ship-only (zero)
            vel_feat,    # 2   shared
            ang_feat,    # 1   ship-only (zero)
            scalars,     # 3   ship-only (zero)
            team_feat,   # 3   shared
            cr_feat,     # 1   shared
            alive_feat,  # 1   ship-only (zero)
            action_feat, # 12  ship-only (zero)
            gc_feat,     # 4n  obstacle-only
            obs_hit,     # 1   obstacle-only
        ], dim=-1)  # (..., M, 12n+24)

        return self.feature_extractor(raw)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Alias for encode_ships — keeps backward compatibility."""
        return self.encode_ships(obs)


# Backward-compatible alias
ShipEncoder = UnifiedEncoder
