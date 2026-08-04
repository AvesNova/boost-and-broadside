"""Entity token encoder for ships and refractive fields.

Encodes each entity (ship or field) into a d_model-dimensional token via
FeatureCoordinator, which extracts and transforms all raw observation channels.

Two layouts are available, selected by ``ModelConfig.encoder_split``:

- shared (default): one MLP over the full channel vector for every token. Simple,
  and guarantees both entity types land in one latent space — but a field token
  spends most of its input width on ship-only channels that are hard zeros.
- split: a per-type first projection over shared+own channels, then a *shared*
  second projection. Each type gets its own read of the raw channels while the
  common output layer keeps the two token spaces commensurable, which matters
  because the spatial layers apply a single W_qkv to both.
"""

import torch
import torch.nn as nn

from boost_and_broadside.config import ModelConfig
from boost_and_broadside.env.observation import YemongObservation
from boost_and_broadside.train.rl.features import FeatureCoordinator, FeatureScope


class BulletEncoder(nn.Module):
    """Projects raw bullet channels into the trunk's token width.

    Deliberately narrow. It runs over N*K entities rather than N+M — 80 against
    12 in the four-field profile — so its hidden width, not the entity encoder's,
    sets encoder cost. A bullet is also a much simpler entity: position,
    velocity, remaining damage and lifetime, plus local field context.

    Its output space is unconstrained because the blocks that read bullets own
    private key/value projections for them, which absorb any change of basis.

    Args:
        model_config: Supplies d_model and bullet_encoder_hidden.
        coordinator: Bullet feature pipeline (``build_bullet_coordinator``).
    """

    def __init__(self, model_config: ModelConfig, coordinator: FeatureCoordinator) -> None:
        super().__init__()
        self.coordinator = coordinator
        hidden = model_config.bullet_encoder_hidden
        self.net = nn.Sequential(
            nn.Linear(coordinator.total_input_dimension, hidden),
            nn.RMSNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, model_config.d_model),
            nn.RMSNorm(model_config.d_model),
        )

    def forward(self, obs: YemongObservation) -> torch.Tensor:
        """Args: obs carrying bullet channels. Returns (..., N*K, d_model)."""
        return self.net(self.coordinator.get_input_vector(obs))


class ShipEncoder(nn.Module):
    """Encodes each entity's raw observations into a d_model-dim token.

    Works on any leading batch shape — the (B, N) dims are treated uniformly.

    Args:
        model_config: Architecture hyperparameters (d_model, encoder_split).
        coordinator: Feature pipeline; drives input dimension and encoding.
        num_ships: N — leading tokens encoded as ships, the rest as fields. Only
            consulted when ``encoder_split`` is set.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        coordinator: FeatureCoordinator,
        num_ships: int | None = None,
    ) -> None:
        super().__init__()
        self.coordinator = coordinator
        self.split = model_config.encoder_split
        self.num_ships = num_ships
        D = model_config.d_model

        if not self.split:
            raw_dim = coordinator.total_input_dimension
            self.feature_extractor = nn.Sequential(
                nn.Linear(raw_dim, 2 * D),
                nn.RMSNorm(2 * D),
                nn.GELU(),
                nn.Linear(2 * D, D),
                nn.RMSNorm(D),
            )
            return

        if num_ships is None:
            raise ValueError("encoder_split requires num_ships to locate the ship/field boundary")

        ship_dim = coordinator.scoped_input_dimension(FeatureScope.SHIP)
        field_dim = coordinator.scoped_input_dimension(FeatureScope.FIELD)
        # Per-type input projection: each entity type reads only its own channels.
        self.ship_proj = nn.Sequential(nn.Linear(ship_dim, 2 * D), nn.RMSNorm(2 * D), nn.GELU())
        self.field_proj = nn.Sequential(nn.Linear(field_dim, 2 * D), nn.RMSNorm(2 * D), nn.GELU())
        # Shared output projection: one latent space for both types.
        self.shared_proj = nn.Sequential(nn.Linear(2 * D, D), nn.RMSNorm(D))

    def forward(self, obs: YemongObservation) -> torch.Tensor:
        """Encode entity observations into tokens.

        Args:
            obs: YemongObservation from YemongEnvWrapper.

        Returns:
            (..., N+M, d_model) float32 token tensor.
        """
        if not self.split:
            raw = self.coordinator.get_input_vector(obs)
            return self.feature_extractor(raw)

        num_ships = self.num_ships
        num_tokens = obs.pos.shape[-2]
        ship_raw = self.coordinator.get_scoped_input_vector(
            obs.slice_tokens(0, num_ships), FeatureScope.SHIP
        )
        hidden = self.ship_proj(ship_raw)

        if num_tokens > num_ships:
            field_raw = self.coordinator.get_scoped_input_vector(
                obs.slice_tokens(num_ships, num_tokens), FeatureScope.FIELD
            )
            hidden = torch.cat([hidden, self.field_proj(field_raw)], dim=-2)

        return self.shared_proj(hidden)
