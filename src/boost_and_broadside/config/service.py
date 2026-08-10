"""Pure services used by the future CLI configuration adapters."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TextIO

from boost_and_broadside.config.fingerprint import canonical_data
from boost_and_broadside.config.schema import RESOLVED_CONFIG_SCHEMA_VERSION, ResolvedTrainConfig


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


def format_resolved_config(
    resolved: ResolvedTrainConfig,
    *,
    launch: Mapping[str, object] | None = None,
) -> str:
    """Format an already resolved launch, so the caller resolves it exactly once."""

    document = resolved_profile_document(resolved)
    if launch is not None:
        document["launch"] = canonical_data(launch)
    rendered = json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return rendered + "\n"


def print_resolved_config(
    resolved: ResolvedTrainConfig,
    *,
    file: TextIO,
    launch: Mapping[str, object] | None = None,
) -> None:
    """Thin output adapter for the ``--print-config`` handler."""

    file.write(format_resolved_config(resolved, launch=launch))
