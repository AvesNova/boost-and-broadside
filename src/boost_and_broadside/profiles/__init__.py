"""Profile registry and resolved projections."""

from __future__ import annotations

from types import MappingProxyType

from boost_and_broadside.config.overrides import apply_overrides
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
    launch_overrides: LaunchOverrides | None = None,
    *,
    overrides: dict[str, str] | None = None,
) -> ResolvedTrainConfig:
    """Resolve a registered profile, with optional ``key=value`` edits applied first.

    Config overrides land on the profile before resolution so that everything
    derived from them -- token width, shard count, normalized discounts -- is
    derived from what was asked for. ``launch_overrides`` is the separate,
    later-applied machine sizing.
    """

    return resolve_profile(named_profile_spec(name, overrides), launch_overrides)


def named_profile_spec(name: str, overrides: dict[str, str] | None = None) -> ProfileSpec:
    """The registered profile after ``key=value`` edits, before any resolution.

    VRAM sizing asks its questions of this rather than of the registered
    profile: an edit to ``num_fields`` or ``num_steps`` changes how much memory
    a launch needs, and a measurement taken without it describes a different
    configuration.
    """

    profile = get_profile(name)
    if overrides:
        profile = apply_overrides(profile, overrides)
    return profile


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
    "named_profile_spec",
    "resolve_named_profile",
]
