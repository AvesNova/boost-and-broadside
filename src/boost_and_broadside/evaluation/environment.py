"""Construction of evaluation physics environments, including field maps."""

import torch

from boost_and_broadside.config import EnvConfig, FieldMapConfig, ShipConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.field_cache import FieldMapCache


def create_evaluation_field_map(
    ship_config: ShipConfig,
    env_config: EnvConfig,
    field_map_config: FieldMapConfig,
    device: str | torch.device,
    *,
    seed: int | None = None,
) -> FieldMapCache:
    """Generate the field distribution declared for an evaluation."""
    if env_config.num_fields <= 0:
        raise ValueError("a field map cannot be created when num_fields is zero")
    return FieldMapCache.generate(
        ship_config, env_config, field_map_config, torch.device(device), seed=seed
    )


def create_evaluation_env(
    num_envs: int,
    ship_config: ShipConfig,
    env_config: EnvConfig,
    device: str | torch.device,
    *,
    field_map_config: FieldMapConfig | None = None,
    field_map: FieldMapCache | None = None,
    field_map_seed: int | None = None,
) -> TensorEnv:
    """Build a ``TensorEnv`` and generate its required field cache when requested.

    Field-bearing evaluations must provide their map-generation intent unless an
    already constructed cache is supplied. Zero-field evaluations remain allocation
    equivalent to constructing ``TensorEnv`` directly.
    """
    torch_device = torch.device(device)
    if env_config.num_fields > 0 and field_map is None:
        if field_map_config is None:
            raise ValueError("field_map_config is required when num_fields > 0")
        field_map = create_evaluation_field_map(
            ship_config,
            env_config,
            field_map_config,
            torch_device,
            seed=field_map_seed,
        )
    return TensorEnv(num_envs, ship_config, env_config, torch_device, field_map)
