"""Checkpoint compatibility: observation contract and policy architecture."""

from collections.abc import Mapping
from typing import Any

import torch.nn as nn

OBSERVATION_SCHEMA = "refractive_fields_v2"

# Keys listed verbatim per mismatch category before the message elides the rest.
_MAX_KEYS_IN_MESSAGE = 6


class IncompatibleCheckpointError(ValueError):
    """Raised when saved weights cannot be loaded into the current model.

    Subclasses ``ValueError`` so callers that only care that the checkpoint is
    unusable can catch either, while the league loader can catch this specific
    type and retire the offending roster entry instead of killing the run.
    """


def require_observation_schema(checkpoint: Mapping[str, Any], path: str | None = None) -> None:
    """Reject weights whose encoder uses a different observation contract.

    The new encoder adds ship-local index and field material inputs, plus a
    local-index auxiliary target. There is no faithful tensor-only migration
    for those learned weights, so failing before model construction gives a
    precise error instead of a long ``load_state_dict`` shape mismatch.

    Raises:
        IncompatibleCheckpointError: If the checkpoint predates, or postdates,
            the current observation contract.
    """

    schema = checkpoint.get("observation_schema")
    if schema == OBSERVATION_SCHEMA:
        return
    location = f" {path!r}" if path is not None else ""
    found = "missing" if schema is None else repr(schema)
    raise IncompatibleCheckpointError(
        f"Checkpoint{location} uses observation schema {found}; expected "
        f"{OBSERVATION_SCHEMA!r}. Observation feature semantics are incompatible "
        "and the policy must be retrained."
    )


def _format_keys(keys: list[str]) -> str:
    """Join keys for an error message, eliding past ``_MAX_KEYS_IN_MESSAGE``."""
    shown = ", ".join(keys[:_MAX_KEYS_IN_MESSAGE])
    remainder = len(keys) - _MAX_KEYS_IN_MESSAGE
    return f"{shown} (+{remainder} more)" if remainder > 0 else shown


def describe_state_dict_mismatch(module: nn.Module, state_dict: Mapping[str, Any]) -> str | None:
    """Summarize how saved weights differ from a module's parameter set.

    The observation schema pins the *input contract*; it says nothing about the
    layers built on top of it. A checkpoint written before or after a submodule
    was added or resized therefore passes ``require_observation_schema`` and
    only fails deep inside ``load_state_dict``, whose message names the keys but
    not the file they came from.

    Args:
        module:     The freshly constructed model the weights are destined for.
        state_dict: The saved parameter mapping.

    Returns:
        A multi-line description of the unexpected, missing, and mis-shaped
        keys, or None when the weights match the module exactly.
    """
    model_state = module.state_dict()
    model_keys = set(model_state)
    saved_keys = set(state_dict)

    unexpected = sorted(saved_keys - model_keys)
    missing = sorted(model_keys - saved_keys)
    mismatched = [
        f"{key} checkpoint {tuple(state_dict[key].shape)} vs model {tuple(model_state[key].shape)}"
        for key in sorted(model_keys & saved_keys)
        if getattr(state_dict[key], "shape", None) != model_state[key].shape
    ]
    if not (unexpected or missing or mismatched):
        return None

    lines = []
    if unexpected:
        lines.append(f"  unexpected in checkpoint ({len(unexpected)}): {_format_keys(unexpected)}")
    if missing:
        lines.append(f"  missing from checkpoint ({len(missing)}): {_format_keys(missing)}")
    if mismatched:
        lines.append(f"  shape mismatch ({len(mismatched)}): {_format_keys(mismatched)}")
    return "\n".join(lines)


def load_policy_weights(
    module: nn.Module, state_dict: Mapping[str, Any], path: str | None = None
) -> None:
    """Load saved weights into a module, naming the file when they do not fit.

    Args:
        module:     Destination model, already constructed.
        state_dict: Saved parameter mapping.
        path:       Checkpoint file the weights came from, used in the error.

    Raises:
        IncompatibleCheckpointError: If the weights were written by a different
            model architecture.
    """
    mismatch = describe_state_dict_mismatch(module, state_dict)
    if mismatch is None:
        module.load_state_dict(state_dict)
        return
    location = f" {path!r}" if path is not None else ""
    raise IncompatibleCheckpointError(
        f"Checkpoint{location} does not match the current "
        f"{type(module).__name__} architecture:\n{mismatch}\n"
        "The weights were written by a different model version — retrain, or "
        "point this run at checkpoints saved by the matching version."
    )
