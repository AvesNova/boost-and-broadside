"""Entity token encoder for ships and refractive fields.

Encodes each entity (ship or field) into a d_model-dimensional token via
FeatureCoordinator, which extracts and transforms all raw observation channels.
"""

import torch
import torch.nn as nn

from boost_and_broadside.config import ModelConfig
from boost_and_broadside.env.observation import YemongObservation
from boost_and_broadside.train.rl.features import FeatureCoordinator


class ShipEncoder(nn.Module):
    """Encodes each entity's raw observations into a d_model-dim token.

    Works on any leading batch shape — the (B, N) dims are treated uniformly.

    Args:
        model_config: Architecture hyperparameters (d_model used here).
        coordinator: Feature pipeline; drives input dimension and encoding.
    """

    def __init__(self, model_config: ModelConfig, coordinator: FeatureCoordinator) -> None:
        super().__init__()
        self.coordinator = coordinator
        raw_dim = coordinator.total_input_dimension

        self.feature_extractor = nn.Sequential(
            nn.Linear(raw_dim, 2 * model_config.d_model),
            nn.RMSNorm(2 * model_config.d_model),
            nn.GELU(),
            nn.Linear(2 * model_config.d_model, model_config.d_model),
            nn.RMSNorm(model_config.d_model),
        )

    def forward(self, obs: YemongObservation) -> torch.Tensor:
        """Encode entity observations into tokens.

        Args:
            obs: YemongObservation from YemongEnvWrapper.

        Returns:
            (..., N, d_model) float32 token tensor.
        """
        raw = self.coordinator.get_input_vector(obs)
        return self.feature_extractor(raw)
