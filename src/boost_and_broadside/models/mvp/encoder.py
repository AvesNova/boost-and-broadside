"""Entity token encoder for ships and obstacles.

Encodes each entity (ship or obstacle) into a d_model-dimensional token.
Ships and obstacles share this encoder.

The encoder is fully driven by ObsConfig: it iterates the feature list,
extracts each source from the obs dict, applies the transform chain, and
concatenates the results into a raw feature vector fed to a two-layer MLP.

Obstacle tokens from the wrapper have:
  - vel: actual obstacle velocity
  - att: velocity direction unit vector (heading = vel_dir for obstacles)
  - ang_vel, prev_*: zeroed
  - health: max_health (full)
  - power, cooldown: 0
  - team_id: 2
"""

import torch
import torch.nn as nn

from boost_and_broadside.config import ModelConfig
from boost_and_broadside.config.obs_spec import ObsConfig


def _extract(obs: dict[str, torch.Tensor], source) -> torch.Tensor:
    """Extract a feature tensor from the obs dict.

    Args:
        obs: Raw obs dict from MVPEnvWrapper._get_obs().
        source: Either a string key (returns the whole tensor) or a
                (key, int) tuple that slices one channel from the last dim,
                removing the trailing dimension.

    Returns:
        Tensor with shape (..., N, D) where D depends on source type.
    """
    if isinstance(source, str):
        return obs[source]
    key, idx = source
    return obs[key][..., idx]  # (..., N) scalar slice


class ShipEncoder(nn.Module):
    """Encodes each entity's raw observations into a d_model-dim token.

    Works on any leading batch shape — the (B, N) dims are treated uniformly.

    Args:
        model_config: Architecture hyperparameters (d_model used here).
        obs_config: Feature pipeline specification; drives input dimension and
                    the encoding logic in forward().
    """

    def __init__(self, model_config: ModelConfig, obs_config: ObsConfig) -> None:
        super().__init__()
        self.obs_config = obs_config
        raw_dim = obs_config.raw_dim()

        self.feature_extractor = nn.Sequential(
            nn.Linear(raw_dim, 2 * model_config.d_model),
            nn.RMSNorm(2 * model_config.d_model),
            nn.GELU(),
            nn.Linear(2 * model_config.d_model, model_config.d_model),
            nn.RMSNorm(model_config.d_model),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode entity observations into tokens.

        Args:
            obs: Raw obs dict from MVPEnvWrapper._get_obs().

        Returns:
            (..., N, d_model) float32 token tensor.
        """
        parts: list[torch.Tensor] = []
        for feat in self.obs_config.features:
            x = _extract(obs, feat.source)
            for t in feat.transforms:
                x = t(x)
            # Scalar sources arrive as (..., N); expand to (..., N, 1) so cat works.
            if x.dim() == obs["team_id"].dim():
                x = x.unsqueeze(-1)
            parts.append(x)

        raw = torch.cat(parts, dim=-1)
        return self.feature_extractor(raw)
