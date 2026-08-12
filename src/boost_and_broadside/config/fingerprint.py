"""Stable canonical serialization and SHA-256 configuration fingerprints."""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import math
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any


def canonical_data(value: Any) -> Any:
    """Convert supported configuration values into deterministic JSON data.

    Runtime schedules are represented by their defining callable and closure,
    matching the S01 characterization format.  Profile intent uses declarative
    schedule specs and therefore does not depend on executable function state.
    """

    if dataclasses.is_dataclass(value):
        return {
            field.name: canonical_data(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if inspect.isfunction(value):
        return {
            "callable": f"{value.__module__}.{value.__qualname__}",
            "closure": {
                name: canonical_data(cell.cell_contents)
                for name, cell in zip(
                    value.__code__.co_freevars,
                    value.__closure__ or (),
                    strict=True,
                )
            },
        }
    if isinstance(value, Enum):
        return {
            "enum": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": canonical_data(value.value),
        }
    if isinstance(value, Mapping):
        return {
            str(key): canonical_data(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [canonical_data(item) for item in value]
    if isinstance(value, (frozenset, set)):
        normalized = [canonical_data(item) for item in value]
        return sorted(normalized, key=canonical_json)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("canonical configuration values must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported canonical configuration value: {type(value).__qualname__}")


def canonical_json(value: Any) -> str:
    """Return compact deterministic JSON with no platform-dependent whitespace."""

    return json.dumps(
        canonical_data(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def fingerprint(value: Any) -> str:
    """Return a stable SHA-256 digest of the canonical serialized value."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
