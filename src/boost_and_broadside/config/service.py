"""Pure services used by the future CLI configuration adapters."""

from __future__ import annotations

import json
from typing import TextIO

from boost_and_broadside.config.fingerprint import canonical_data
from boost_and_broadside.config.resolve import LaunchOverrides
from boost_and_broadside.config.schema import RESOLVED_CONFIG_SCHEMA_VERSION, ResolvedTrainConfig
from boost_and_broadside.profiles import resolve_named_profile


def resolved_profile_document(resolved: ResolvedTrainConfig) -> dict:
    """Return the complete plain-data document stored or printed by adapters."""

    train_config = canonical_data(resolved.train_config)
    train_config["schedule"] = canonical_data(resolved.schedule_spec)
    return {
        "schema_version": RESOLVED_CONFIG_SCHEMA_VERSION,
        "profile": resolved.profile_name,
        "profile_fingerprint": resolved.profile_fingerprint,
        "resolved_config_fingerprint": resolved.resolved_config_fingerprint,
        "config": {
            "ship_config": canonical_data(resolved.ship_config),
            "model_config": canonical_data(resolved.model_config),
            "train_config": train_config,
        },
        "sources": dict(sorted(resolved.value_sources.items())),
    }


def format_resolved_profile(
    name: str,
    overrides: LaunchOverrides | None = None,
) -> str:
    """Resolve, validate, fingerprint, and format a launch without allocation."""

    document = resolved_profile_document(resolve_named_profile(name, overrides))
    rendered = json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return rendered + "\n"


def print_resolved_profile(
    name: str,
    overrides: LaunchOverrides | None = None,
    *,
    file: TextIO,
) -> None:
    """Thin output adapter for S05's eventual ``--print-config`` handler."""

    file.write(format_resolved_profile(name, overrides))
