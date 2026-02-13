"""
Observation tokenization for neural network input.

Converts raw game observations into normalized token representations suitable for
transformer-based agents.
"""

import torch

from boost_and_broadside.core.constants import (
    StateFeature,
    STATE_DIM,
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
        Token tensor of shape (1, num_ships, 5).
        Features include (via StateFeature enum):
        - HEALTH (normalized)                                       [0]
        - POWER (normalized)                                        [1]
        - VX (normalized)                                           [2]
        - VY (normalized)                                           [3]
        - ANG_VEL (normalized)                                       [4]
    """
    # Allocate tokens based on global STATE_DIM
    batch_size = 1
    num_ships = obs["health"].shape[0]
    tokens = torch.zeros((batch_size, num_ships, STATE_DIM), device=obs["health"].device)

    # Fill features according to StateFeature enum
    tokens[..., StateFeature.HEALTH] = obs["health"].float()
    tokens[..., StateFeature.POWER] = obs["power"].float()
    
    # Handle velocity (complex -> real/imag)
    tokens[..., StateFeature.VX] = obs["velocity"].real.float()
    tokens[..., StateFeature.VY] = obs["velocity"].imag.float()
    
    tokens[..., StateFeature.ANG_VEL] = obs["angular_velocity"].float()

    return tokens
