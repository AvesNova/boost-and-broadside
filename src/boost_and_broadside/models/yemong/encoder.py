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
from boost_and_broadside.env.observation import ObjectType, ObsKey, YemongObservation
from boost_and_broadside.train.rl.features import FeatureCoordinator, FeatureScope


class BulletEncoder(nn.Module):
    """Projects raw bullet channels into the trunk's token width.

    Deliberately narrow. It runs over N*K entities rather than N+M — 80 against
    12 in the four-field profile — so its hidden width, not the entity encoder's,
    sets encoder cost. A bullet is also a much simpler entity: position,
    velocity and lifetime, plus local field context.

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

        scopes = {
            ObjectType.SHIP: FeatureScope.SHIP,
            ObjectType.FIELD: FeatureScope.FIELD,
            ObjectType.ZONE: FeatureScope.ZONE,
            ObjectType.BOUNDARY: FeatureScope.BOUNDARY,
        }
        self._type_scopes = scopes
        self.type_proj = nn.ModuleDict(
            {
                str(int(object_type)): nn.Sequential(
                    nn.Linear(coordinator.scoped_input_dimension(scope), 2 * D),
                    nn.RMSNorm(2 * D),
                    nn.GELU(),
                )
                for object_type, scope in scopes.items()
            }
        )
        # Shared output projection: one latent space for both types.
        self.shared_proj = nn.Sequential(nn.Linear(2 * D, D), nn.RMSNorm(D))
        # Token-axis spans per object type, derived on first use and cached under
        # the token count. Not a parameter and not in the state dict: no weight
        # is sized by any token count, so a checkpoint stays portable between
        # fleet and map sizes and simply re-derives on its first forward.
        self._span_key: int | None = None
        self._spans: tuple[tuple[int, int, int], ...] = ()

    def _token_spans(self, obs: YemongObservation) -> tuple[tuple[int, int, int], ...]:
        """``(object_type, start, end)`` per contiguous run of the token axis.

        The environment lays the axis out grouped by kind -- ships, then fields,
        then zones, then the boundary token -- identically for every env in the
        batch, so one run per kind describes it. Derived from the observation
        rather than from a configured count, which is what keeps the encoder
        agnostic to how many of each kind exist.

        Cached under the token count. Reading the types is a device sync, so it
        must not happen per forward; the counts are fixed for a run, and a policy
        moved to a differently-shaped environment re-derives on its next call.
        """

        object_types = obs[ObsKey.OBJECT_TYPE]
        tokens = int(object_types.shape[-1])
        if self._span_key == tokens:
            return self._spans

        row = object_types.reshape(-1, tokens)[0].tolist()
        spans: list[tuple[int, int, int]] = []
        for index, kind in enumerate(row):
            if spans and spans[-1][0] == kind:
                spans[-1] = (kind, spans[-1][1], index + 1)
            else:
                spans.append((int(kind), index, index + 1))
        seen = [kind for kind, _, _ in spans]
        if len(seen) != len(set(seen)):
            raise ValueError(
                f"encoder_split needs the token axis grouped by object type, got runs {seen}"
            )
        self._span_key, self._spans = tokens, tuple(spans)
        return self._spans

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

        # One projection per kind, over that kind's tokens only. The previous
        # form ran all four over all N+M tokens and discarded three quarters of
        # the result with ``torch.where`` -- and encoded four times as much as it
        # used, since the scoped input vector was built for the whole axis too.
        parts = []
        for object_type, start, end in self._token_spans(obs):
            scope = self._type_scopes[ObjectType(object_type)]
            raw = self.coordinator.get_scoped_input_vector(obs.slice_tokens(start, end), scope)
            parts.append(self.type_proj[str(int(object_type))](raw))
        hidden = torch.cat(parts, dim=-2)
        return self.shared_proj(hidden)

    @property
    def ship_proj(self) -> nn.Sequential:
        """Compatibility name for the ship type's first-stage projection."""

        return self.type_proj[str(int(ObjectType.SHIP))]

    @property
    def field_proj(self) -> nn.Sequential:
        """Compatibility name for the field type's first-stage projection."""

        return self.type_proj[str(int(ObjectType.FIELD))]
