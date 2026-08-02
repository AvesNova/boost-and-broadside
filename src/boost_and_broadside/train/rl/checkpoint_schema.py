"""Checkpoint compatibility for the refractive-field observation contract."""

from collections.abc import Mapping
from typing import Any

OBSERVATION_SCHEMA = "refractive_fields_v3"


def require_observation_schema(checkpoint: Mapping[str, Any], path: str | None = None) -> None:
    """Reject weights whose encoder uses a different observation contract.

    v2 added ship-local index and field material inputs plus a local-index
    auxiliary target. v3 adds the ship's local grad(n) channels, which widen the
    encoder's first projection. There is no faithful tensor-only migration for
    those learned weights, so failing before model construction gives a precise
    error instead of a long ``load_state_dict`` shape mismatch.

    Trunk-structure changes are deliberately *not* gated here: the Yemong block
    rename and the ship/field sublayer split are pure state-dict key remaps, and
    a mismatched ``ModelConfig`` surfaces as an ordinary load error.
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
