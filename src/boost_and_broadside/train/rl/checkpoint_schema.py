"""Checkpoint compatibility for the explicit policy-observation contract."""

import math
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

OBSERVATION_SCHEMA = "frontline_world_v4"
POSITION_FINEST_PERIOD = 128.0


def position_fourier_frequencies(period: float) -> int:
    """Base-2 frequencies needed to keep the finest period at most 128 px."""

    return max(1, math.ceil(math.log2(period / POSITION_FINEST_PERIOD)) + 1)


def observation_contract(ship_config: Any) -> dict[str, Any]:
    """Small explicit descriptor of learned feature layout and semantics."""

    world_size = (
        ship_config["world_size"] if isinstance(ship_config, Mapping) else ship_config.world_size
    )
    return {
        "version": 4,
        "position_fourier_basis": "base2",
        "position_finest_period": POSITION_FINEST_PERIOD,
        "position_frequencies": tuple(
            position_fourier_frequencies(float(period)) for period in world_size
        ),
    }


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

    v4 makes the world-size-dependent base-2 position-frequency count explicit;
    1024 uses four frequencies and the 16384 frontline world uses eight, both
    preserving an approximately 128 px finest period. There is no faithful
    tensor-only migration for the widened learned projection.

    Trunk-structure changes are deliberately *not* gated here: the Yemong block
    rename and the ship/field sublayer split are pure state-dict key remaps, and
    a mismatched ``ModelConfig`` surfaces as an ordinary load error.
    """

    schema = checkpoint.get("observation_schema")
    location = f" {path!r}" if path is not None else ""
    if schema != OBSERVATION_SCHEMA:
        found = "missing" if schema is None else repr(schema)
        raise ValueError(
            f"Checkpoint{location} uses observation schema {found}; expected "
            f"{OBSERVATION_SCHEMA!r}. Observation feature semantics are incompatible "
            "and the policy must be retrained."
        )
    ship_config = checkpoint.get("ship_config")
    if not isinstance(ship_config, Mapping) or "world_size" not in ship_config:
        raise ValueError(
            f"Checkpoint{location} is missing ship_config.world_size required by "
            "the observation contract."
        )
    expected = observation_contract(ship_config)
    if checkpoint.get("observation_contract") != expected:
        raise ValueError(
            f"Checkpoint{location} has an incompatible observation contract; "
            f"expected {expected!r}, found {checkpoint.get('observation_contract')!r}."
        )
