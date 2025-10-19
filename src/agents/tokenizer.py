from numpy import dtype
import torch


def observation_to_tokens(obs: dict, perspective: int) -> torch.Tensor:
    # Normalized positions
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
            # obs["ship_id"].float() / 8,
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
