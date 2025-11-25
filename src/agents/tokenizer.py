"""
Observation tokenization for neural network input.

Converts raw game observations into normalized token representations suitable for
transformer-based agents.
"""
import torch


def observation_to_tokens(obs: dict[str, torch.Tensor], perspective: int) -> torch.Tensor:
    """
    Convert observation dictionary to token representation.

    Transforms raw game state into a normalized feature vector suitable for
    neural network processing. Uses sin/cos encoding for positional features
    to handle toroidal world wrapping.

    Args:
        obs: Observation dictionary containing ship states.
        perspective: Team ID from whose perspective to encode (0 or 1).

    Returns:
        Token tensor of shape (1, num_ships, token_dim) where token_dim=12.
        Features include:
        - Team indicator (binary)
        - Health (normalized)
        - Power (normalized)
        - Position (sin/cos encoded x and y)
        - Velocity (normalized x and y components)
        - Attitude (x and y components)
        - Shooting state (binary)
    """
    # Normalized positions (hardcoded world size: 1200x800)
    x_norm = obs["position"].real / 1200
    y_norm = obs["position"].imag / 800

    # Wrap-around encoding using sin/cos
    x_sin = torch.sin(2 * torch.pi * x_norm)
    x_cos = torch.cos(2 * torch.pi * x_norm)
    y_sin = torch.sin(2 * torch.pi * y_norm)
    y_cos = torch.cos(2 * torch.pi * y_norm)

    # Stack features with sin/cos encoding for positions
    return torch.stack(
        [
            torch.eq(obs["team_id"], perspective),
            obs["health"].float() / 100,
            obs["power"].float() / 100,
            x_sin,
            x_cos,
            y_sin,
            y_cos,
            obs["velocity"].real / 180.0,
            obs["velocity"].imag / 180.0,
            obs["attitude"].real,
            obs["attitude"].imag,
            obs["is_shooting"].float(),
        ],
        dim=1,
    ).unsqueeze(0)
