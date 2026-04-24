"""Entity token encoder for ships and obstacles.

Encodes each entity (ship or obstacle) into a d_model-dimensional token.
Ships and obstacles share this encoder. Obstacle tokens have ship-specific
features zeroed out by the wrapper; only pos, radius, alive, and team_id=2
carry meaningful values for obstacles.

Feature breakdown (per entity, n_freqs=8 example):
    Fourier position: n_freqs x 2 (sin+cos) x 2 (x, y) = 4*n_freqs dims
    Fourier attitude: n_freqs x 2 (sin+cos) x 2 (nose, vel_dir) = 4*n_freqs dims
    Symlog velocity:  (vx, vy)                          =  2 dims
    Symlog ang_vel:   scalar                             =  1 dim
    Scalars:          (health, power, cooldown) normed   =  3 dims
    Team embedding:   nn.Embedding(3, 4) (2=obstacle)   =  4 dims
    Alive:            scalar                             =  1 dim
    Prev-action one-hot: 3 + 7 + 2                      = 12 dims
    Radius:           normalized scalar                  =  1 dim
    ---
    Total raw dim (n_freqs=8)                            = 88 dims
    -> Linear(raw_dim, d_model)
"""

import torch
import torch.nn as nn

from boost_and_broadside.config import ModelConfig, ShipConfig
from boost_and_broadside.constants import (
    NUM_POWER_ACTIONS,
    NUM_TURN_ACTIONS,
    NUM_SHOOT_ACTIONS,
)


_TEAM_EMBED_DIM = 4
_ACTION_DIM = NUM_POWER_ACTIONS + NUM_TURN_ACTIONS + NUM_SHOOT_ACTIONS  # 12


def _raw_dim(n_fourier_freqs: int) -> int:
    """Compute raw feature dimension given number of Fourier frequencies."""
    pos_dim = 4 * n_fourier_freqs  # x and y, each (sin+cos) * n_freqs
    att_dim = 4 * n_fourier_freqs  # nose_att and vel_att, each (sin+cos) * n_freqs
    return pos_dim + att_dim + 2 + 1 + 3 + _TEAM_EMBED_DIM + 1 + _ACTION_DIM + 1  # +1 radius


def _symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric log: sign(x) * log(|x| + 1).  Compresses large values."""
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def _fourier_encode(
    coords: torch.Tensor, n_freqs: int, max_period: float
) -> torch.Tensor:
    """Fourier-encode a batch of scalar coordinates using base-2 power frequencies.

    Args:
        coords: (...) float tensor — coordinate values.
        n_freqs: Number of frequency bands.
        max_period: The largest wavelength (= world_size for position).

    Returns:
        (..., 2 * n_freqs) tensor of [sin, cos] features.
    """
    # Standard NeRF-style positional encoding: 2^0, 2^1, ..., 2^{n_freqs-1}
    k = torch.arange(n_freqs, device=coords.device, dtype=torch.float32)
    freqs = (2.0 * torch.pi / max_period) * (2.0**k)  # (n_freqs,)

    args = coords.unsqueeze(-1) * freqs  # (..., n_freqs)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (..., 2*n_freqs)


def _fourier_encode_angle(cossin: torch.Tensor, n_freqs: int) -> torch.Tensor:
    """Fourier-encode a unit direction vector with base-2 integer frequencies to preserve 2π circularity.

    Args:
        cossin: (..., 2) float tensor of [cos, sin] pairs.
        n_freqs: Number of frequency bands.

    Returns:
        (..., 2 * n_freqs) tensor of [sin, cos] features.
    """
    angle = torch.atan2(cossin[..., 1], cossin[..., 0])  # (...)

    # Base-2 integer frequencies: 2^0, 2^1, ..., 2^{n_freqs-1}
    # Since these are all integers, it perfectly wraps at 2pi.
    k = torch.arange(n_freqs, device=angle.device, dtype=torch.float32)
    freqs = 2.0**k

    args = angle.unsqueeze(-1) * freqs  # (..., n_freqs)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (..., 2*n_freqs)


class ShipEncoder(nn.Module):
    """Encodes each ship's raw observations into a d_model-dim token.

    Works on any leading batch shape — the (B, N) dims are treated uniformly.
    """

    def __init__(self, model_config: ModelConfig, ship_config: ShipConfig) -> None:
        super().__init__()
        self.n_freqs = model_config.n_fourier_freqs
        self.world_w, self.world_h = ship_config.world_size
        self.d_model = model_config.d_model

        self.team_embed = nn.Embedding(3, _TEAM_EMBED_DIM)  # 0/1=ship teams, 2=obstacle
        self.feature_extractor = nn.Sequential(
            nn.Linear(_raw_dim(self.n_freqs), 2 * model_config.d_model),
            nn.RMSNorm(2 * model_config.d_model),
            nn.GELU(),
            nn.Linear(2 * model_config.d_model, model_config.d_model),
            nn.RMSNorm(model_config.d_model),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode entity observations into tokens.

        Works on combined (ship + obstacle) token sequences. Obstacle tokens
        have vel/att/ang_vel/scalars/prev_action zeroed by the wrapper.

        Args:
            obs: Dict with keys matching MVPEnvWrapper._get_obs():
                "pos"         (..., N, 2) float32 — normalized [0,1]
                "vel"         (..., N, 2) float32
                "att"         (..., N, 2) float32
                "ang_vel"     (..., N, 1) float32
                "scalars"     (..., N, 3) float32
                "team_id"     (..., N)   int32  (2 for obstacles)
                "alive"       (..., N)   bool
                "prev_action" (..., N, 3) int64
                "radius"      (..., N, 1) float32 — normalized by obstacle_radius_max

        Returns:
            (..., N, d_model) float32 token tensor.
        """
        pos = obs["pos"]  # (..., N, 2)
        vel = obs["vel"]  # (..., N, 2)

        speed = torch.norm(vel, dim=-1, keepdim=True)
        speed_safe = torch.clamp(speed, min=1e-6)
        vel_dir = vel / speed_safe  # (..., N, 2)

        att = obs["att"]  # (..., N, 2)
        ang_vel = obs["ang_vel"]  # (..., N, 1)
        scalars = obs["scalars"]  # (..., N, 3)
        team_id = obs["team_id"].long()  # (..., N)
        alive = obs["alive"]  # (..., N)
        prev_action = obs["prev_action"].long()  # (..., N, 3)

        # Fourier position encoding — encode x and y separately
        px_enc = _fourier_encode(pos[..., 0], self.n_freqs, 1.0)  # (..., N, 2*n_freqs)
        py_enc = _fourier_encode(pos[..., 1], self.n_freqs, 1.0)  # (..., N, 2*n_freqs)
        pos_feat = torch.cat([px_enc, py_enc], dim=-1)  # (..., N, 4*n_freqs)

        # Fourier attitude encoding (circular, uses integer frequencies)
        nose_att_enc = _fourier_encode_angle(att, self.n_freqs)  # (..., N, 2*n_freqs)
        vel_att_enc = _fourier_encode_angle(
            vel_dir, self.n_freqs
        )  # (..., N, 2*n_freqs)
        att_feat = torch.cat([nose_att_enc, vel_att_enc], dim=-1)  # (..., N, 4*n_freqs)

        # Symlog velocity and angular velocity
        vel_feat = _symlog(vel)  # (..., N, 2)
        ang_feat = _symlog(ang_vel)  # (..., N, 1)

        # Team embedding
        team_feat = self.team_embed(team_id)  # (..., N, 4)

        # Alive as float
        alive_feat = alive.float().unsqueeze(-1)  # (..., N, 1)

        # Prev-action one-hot — concatenate one-hot for each sub-action
        pa_power = torch.nn.functional.one_hot(
            prev_action[..., 0], NUM_POWER_ACTIONS
        ).float()
        pa_turn = torch.nn.functional.one_hot(
            prev_action[..., 1], NUM_TURN_ACTIONS
        ).float()
        pa_shoot = torch.nn.functional.one_hot(
            prev_action[..., 2], NUM_SHOOT_ACTIONS
        ).float()
        action_feat = torch.cat([pa_power, pa_turn, pa_shoot], dim=-1)  # (..., N, 12)

        radius = obs["radius"]  # (..., N, 1) normalized

        raw = torch.cat(
            [
                pos_feat,   # 4 * n_freqs
                att_feat,   # 4 * n_freqs
                vel_feat,   # 2
                ang_feat,   # 1
                scalars,    # 3
                team_feat,  # 4
                alive_feat,  # 1
                action_feat,  # 12
                radius,     # 1
            ],
            dim=-1,
        )  # (..., N, 24 + 8*n_freqs)

        return self.feature_extractor(raw)  # (..., N, d_model)
