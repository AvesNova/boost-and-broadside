"""Public re-exports for the boost_and_broadside.config package."""

from boost_and_broadside.config.core import (
    EnvConfig,
    InterfaceDamageLevel,
    ModelConfig,
    RefractiveIndexLevel,
    RewardConfig,
    ShipConfig,
)
from boost_and_broadside.config.fingerprint import canonical_data, canonical_json, fingerprint
from boost_and_broadside.config.resolve import (
    LaunchOverrides,
    derive_aligned_num_envs,
    derive_time_normalized_value,
    resolve_profile,
    validate_resolved_config,
)
from boost_and_broadside.config.schedule import (
    Schedule,
    TrainingSchedule,
    constant,
    cosine_anneal,
    exponential,
    join,
    linear,
    stepped,
)
from boost_and_broadside.config.schedule_spec import (
    Keypoints,
    TrainingScheduleSpec,
    compile_keypoints,
    hold,
)
from boost_and_broadside.config.schema import (
    PROFILE_SCHEMA_VERSION,
    RESOLVED_CONFIG_SCHEMA_VERSION,
    LaunchSizingSpec,
    ProfileSpec,
    ResolvedTrainConfig,
)
from boost_and_broadside.config.training import (
    EloCalibrateConfig,
    EloEvalConfig,
    FieldMapConfig,
    ObstacleCacheConfig,
    ScaleConfig,
    TrainConfig,
)

__all__ = [
    "ShipConfig",
    "EnvConfig",
    "RefractiveIndexLevel",
    "InterfaceDamageLevel",
    "EloCalibrateConfig",
    "EloEvalConfig",
    "ModelConfig",
    "RewardConfig",
    "Schedule",
    "TrainingSchedule",
    "constant",
    "linear",
    "stepped",
    "exponential",
    "cosine_anneal",
    "join",
    "ObstacleCacheConfig",
    "FieldMapConfig",
    "ScaleConfig",
    "TrainConfig",
    "LaunchOverrides",
    "LaunchSizingSpec",
    "ProfileSpec",
    "PROFILE_SCHEMA_VERSION",
    "RESOLVED_CONFIG_SCHEMA_VERSION",
    "ResolvedTrainConfig",
    "TrainingScheduleSpec",
    "Keypoints",
    "compile_keypoints",
    "hold",
    "canonical_data",
    "canonical_json",
    "derive_aligned_num_envs",
    "derive_time_normalized_value",
    "fingerprint",
    "resolve_profile",
    "validate_resolved_config",
]
