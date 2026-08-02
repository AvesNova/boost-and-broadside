"""Checkpoint compatibility for the refractive-field observation contract."""

from collections.abc import Mapping
from typing import Any

OBSERVATION_SCHEMA = "refractive_fields_v2"


def require_observation_schema(checkpoint: Mapping[str, Any], path: str | None = None) -> None:
    """Reject weights whose encoder uses a different observation contract.

    The new encoder adds ship-local index and field material inputs, plus a
    local-index auxiliary target. There is no faithful tensor-only migration
    for those learned weights, so failing before model construction gives a
    precise error instead of a long ``load_state_dict`` shape mismatch.
    """

    schema = checkpoint.get("observation_schema")
    if schema == OBSERVATION_SCHEMA:
        return
    location = f" {path!r}" if path is not None else ""
    found = "missing" if schema is None else repr(schema)
    raise ValueError(
        f"Checkpoint{location} uses observation schema {found}; expected "
        f"{OBSERVATION_SCHEMA!r}. Observation feature semantics are incompatible "
        "and the policy must be retrained."
    )
