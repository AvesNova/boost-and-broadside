"""Checkpoint compatibility for the refractive-field observation contract."""

import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

OBSERVATION_SCHEMA = "refractive_fields_v3"


def load_checkpoint_payload(
    path: str | Path,
    *,
    map_location: str | torch.device,
) -> Mapping[str, Any]:
    """Read a checkpoint and normalize expected corrupt-input failures."""

    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except (EOFError, OSError, pickle.UnpicklingError, RuntimeError) as error:
        raise ValueError(
            f"could not read checkpoint {str(path)!r}: {type(error).__name__}: {error}"
        ) from None
    if not isinstance(checkpoint, Mapping):
        raise ValueError(
            f"could not read checkpoint {str(path)!r}: expected a mapping payload, "
            f"got {type(checkpoint).__name__}"
        )
    return checkpoint


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
