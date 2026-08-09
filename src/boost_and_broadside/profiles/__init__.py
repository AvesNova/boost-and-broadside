"""Independent profile registry and resolved legacy projections."""

from __future__ import annotations

from types import MappingProxyType

from boost_and_broadside.config.resolve import LaunchOverrides, resolve_profile
from boost_and_broadside.config.schema import ProfileSpec, ResolvedTrainConfig
from boost_and_broadside.profiles.bc import BC_PROFILE
from boost_and_broadside.profiles.legacy_bc_warmstart import (
    LEGACY_BC_WARMSTART_PRETRAIN_PROFILE,
)
from boost_and_broadside.profiles.rl import RL_PROFILE
from boost_and_broadside.profiles.rl_fields import RL_FIELDS_PROFILE

PROFILES = MappingProxyType(
    {
        "rl": RL_PROFILE,
        "rl-fields": RL_FIELDS_PROFILE,
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


# S02 keeps trainer consumers field-compatible while moving construction behind
# the new resolver.  Later CLI/training sections consume the resolved wrapper.
RL_RESOLVED_CONFIG = resolve_named_profile("rl")
RL_FIELDS_RESOLVED_CONFIG = resolve_named_profile("rl-fields")
BC_RESOLVED_CONFIG = resolve_named_profile("bc")

RL_TRAIN_CONFIG = RL_RESOLVED_CONFIG.train_config
RL_FIELDS_TRAIN_CONFIG = RL_FIELDS_RESOLVED_CONFIG.train_config
BC_TRAIN_CONFIG = BC_RESOLVED_CONFIG.train_config

BC_WARMSTART_PRETRAIN_RESOLVED_CONFIG = resolve_profile(LEGACY_BC_WARMSTART_PRETRAIN_PROFILE)
BC_WARMSTART_PRETRAIN_CONFIG = BC_WARMSTART_PRETRAIN_RESOLVED_CONFIG.train_config
BC_WARMSTART_RL_CONFIG = RL_TRAIN_CONFIG

__all__ = [
    "BC_PROFILE",
    "BC_RESOLVED_CONFIG",
    "BC_TRAIN_CONFIG",
    "BC_WARMSTART_PRETRAIN_CONFIG",
    "BC_WARMSTART_PRETRAIN_RESOLVED_CONFIG",
    "BC_WARMSTART_RL_CONFIG",
    "PROFILES",
    "RL_FIELDS_PROFILE",
    "RL_FIELDS_RESOLVED_CONFIG",
    "RL_FIELDS_TRAIN_CONFIG",
    "RL_PROFILE",
    "RL_RESOLVED_CONFIG",
    "RL_TRAIN_CONFIG",
    "get_profile",
    "resolve_named_profile",
]
