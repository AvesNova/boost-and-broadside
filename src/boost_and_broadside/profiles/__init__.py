"""Profile registry and resolved projections."""

from __future__ import annotations

from types import MappingProxyType

from boost_and_broadside.config.resolve import LaunchOverrides, resolve_profile
from boost_and_broadside.config.schema import ProfileSpec, ResolvedTrainConfig
from boost_and_broadside.profiles.bc import BC_PROFILE
from boost_and_broadside.profiles.rl import RL_PROFILE

PROFILES = MappingProxyType(
    {
        "rl": RL_PROFILE,
        "bc": BC_PROFILE,
    }
)


def get_profile(name: str) -> ProfileSpec:
    """Return an exact registered profile name."""

    try:
        return PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(PROFILES))
        raise KeyError(f"unknown profile {name!r}; choose one of: {choices}") from exc


def resolve_named_profile(
    name: str,
    overrides: LaunchOverrides | None = None,
) -> ResolvedTrainConfig:
    return resolve_profile(get_profile(name), overrides)


RL_RESOLVED_CONFIG = resolve_named_profile("rl")
BC_RESOLVED_CONFIG = resolve_named_profile("bc")

RL_TRAIN_CONFIG = RL_RESOLVED_CONFIG.train_config
BC_TRAIN_CONFIG = BC_RESOLVED_CONFIG.train_config

__all__ = [
    "BC_PROFILE",
    "BC_RESOLVED_CONFIG",
    "BC_TRAIN_CONFIG",
    "PROFILES",
    "RL_PROFILE",
    "RL_RESOLVED_CONFIG",
    "RL_TRAIN_CONFIG",
    "get_profile",
    "resolve_named_profile",
]
