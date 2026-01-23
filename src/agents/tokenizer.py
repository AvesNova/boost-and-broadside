"""
Observation tokenization for neural network input.

Converts raw game observations into normalized token representations suitable for
transformer-based agents.
"""

import torch

from core.constants import (
    NORM_VELOCITY,
    NORM_ACCELERATION,
    NORM_ANGULAR_VELOCITY,
    NORM_HEALTH,
    NORM_POWER,
)


def observation_to_tokens(
    obs: dict[str, torch.Tensor], perspective: int, world_size: tuple[float, float]
) -> torch.Tensor:
    """
    Convert observation dictionary to token representation.

    Transforms raw game state into a normalized feature vector suitable for
    neural network processing. Uses sin/cos encoding for positional features
    to handle toroidal world wrapping.

    Args:
        obs: Observation dictionary containing ship states.
        perspective: Team ID from whose perspective to encode (0 or 1).
        world_size: Dimensions of the world (width, height) for normalization.

    Returns:
        Token tensor of shape (1, num_ships, token_dim) where token_dim=15.
        Features include:
        - Team indicator (binary)                                   [0]
        - Health (normalized)                                       [1]
        - Power (normalized)                                        [2]
        - Position (sin/cos encoded x and y)                        [3, 4, 5, 6]
        - Velocity (normalized x and y components)                  [7, 8]
        - Attitude (x and y components)                             [9, 10]
        - Shooting state (binary)                                   [11]
        - Acceleration (normalized x and y components)              [12, 13]
        - Angular Velocity (normalized)                             [14]
    """
    # Normalized positions
    x_norm = obs["position"].real / world_size[0]
    y_norm = obs["position"].imag / world_size[1]

    # Wrap-around encoding using sin/cos
    x_sin = torch.sin(2 * torch.pi * x_norm)
    x_cos = torch.cos(2 * torch.pi * x_norm)
    y_sin = torch.sin(2 * torch.pi * y_norm)
    y_cos = torch.cos(2 * torch.pi * y_norm)

    # Stack features with sin/cos encoding for positions
    return torch.stack(
        [
            torch.eq(obs["team_id"], perspective),
            obs["health"].float() / NORM_HEALTH,
            obs["power"].float() / NORM_POWER,
            x_sin,
            x_cos,
            y_sin,
            y_cos,
            obs["velocity"].real / NORM_VELOCITY,
            obs["velocity"].imag / NORM_VELOCITY,
            obs["attitude"].real,
            obs["attitude"].imag,
            obs["is_shooting"].float(),
            obs["acceleration"].real / NORM_ACCELERATION, # Approximate max accel
            obs["acceleration"].imag / NORM_ACCELERATION,
            obs["angular_velocity"] / NORM_ANGULAR_VELOCITY,  # Approximate max ang vel
        ],
        dim=1,
    ).unsqueeze(0)
