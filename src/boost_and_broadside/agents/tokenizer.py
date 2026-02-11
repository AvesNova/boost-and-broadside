"""
Observation tokenization for neural network input.

Converts raw game observations into normalized token representations suitable for
transformer-based agents.
"""

import torch

from boost_and_broadside.core.constants import (
    NORM_VELOCITY,
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
        Token tensor of shape (1, num_ships, token_dim) where token_dim=9.
        Features include:
        - Team indicator (binary)                                   [0]
        - Health (normalized)                                       [1]
        - Power (normalized)                                        [2]
        - Velocity (normalized x and y components)                  [3, 4]
        - Attitude (x and y components)                             [5, 6]
        - Shooting state (binary)                                   [7]
        - Angular Velocity (normalized)                             [8]
    """
    # Stack features
    return torch.stack(
        [
            torch.eq(obs["team_id"], perspective),
            obs["health"].float() / NORM_HEALTH,
            obs["power"].float() / NORM_POWER,
            obs["velocity"].real / NORM_VELOCITY,
            obs["velocity"].imag / NORM_VELOCITY,
            obs["attitude"].real,
            obs["attitude"].imag,
            obs["is_shooting"].float(),
            obs["angular_velocity"] / NORM_ANGULAR_VELOCITY,
        ],
        dim=1,
    ).unsqueeze(0)
