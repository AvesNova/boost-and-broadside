from collections.abc import Iterable
from dataclasses import replace

import torch

from boost_and_broadside.config import EnvConfig, FieldMapConfig, ShipConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.field_cache import FieldMapCache
from boost_and_broadside.evaluation.agents import ResolvedAgent


def resolve_evaluation_environment(
    env_config: EnvConfig, agents: Iterable[ResolvedAgent]
) -> tuple[EnvConfig, FieldMapConfig | None]:
    """Resolve a shared field distribution from policy checkpoint provenance.

    Random and scripted agents have no environment provenance. Every field
    policy in one evaluation must declare the same field count and map
    distribution; otherwise the requested matchup has no faithful shared task.
    """

    declarations = [
        (agent.bundle.env_config, agent.bundle.field_map_config)
        for agent in agents
        if agent.kind == "policy"
        and agent.bundle is not None
        and agent.bundle.env_config is not None
    ]
    field_declarations = [
        (candidate, field_map)
        for candidate, field_map in declarations
        if candidate.num_fields > 0
    ]
    if not field_declarations:
        return env_config, None

    selected_env, selected_map = field_declarations[0]
    if selected_map is None:
        raise ValueError(
            "field checkpoint does not record field-map intent; cannot evaluate it faithfully"
        )
    for candidate_env, candidate_map in field_declarations[1:]:
        if candidate_env.num_fields != selected_env.num_fields or candidate_map != selected_map:
            raise ValueError("field checkpoints declare incompatible evaluation environments")
    return replace(env_config, num_fields=selected_env.num_fields), selected_map


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


def run_field_map(
    ship_config: ShipConfig,
    env_config: EnvConfig,
    field_map_config: FieldMapConfig | None,
    device: str | torch.device,
) -> FieldMapCache | None:
    """Regenerate the map distribution a fields run trained on, or nothing.

    Evaluation modes that rate a finished run read the run's own field-map
    intent rather than a profile's: the run is the subject, and the maps it
    trained on are part of what is being rated.
    """

    if field_map_config is None or env_config.num_fields <= 0:
        return None
    print(f"  generating field map cache ({field_map_config.cache_size} maps)...")
    return create_evaluation_field_map(ship_config, env_config, field_map_config, device)


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
