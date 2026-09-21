"""Checkpoint compatibility for the explicit policy-observation contract."""

import math
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

OBSERVATION_SCHEMA = "frontline_shields_v9"
POSITION_FINEST_PERIOD = 128.0
# Harmonics the attitude Fourier feature expands the heading angle on. Defined
# here, beside the position count, because rotary spatial attention reuses both
# bases verbatim and a second definition of either would let the input features
# and the Q/K rotations drift apart silently.
ATTITUDE_FOURIER_FREQUENCIES = 4


def position_fourier_frequencies(period: float) -> int:
    """Base-2 frequencies needed to keep the finest period at most 128 px."""

    return max(1, math.ceil(math.log2(period / POSITION_FINEST_PERIOD)) + 1)


def base2_frequencies(period: float, n_freqs: int) -> tuple[float, ...]:
    """Angular frequencies of a base-2 Fourier expansion over ``period``.

    The single definition of ``(2*pi / period) * 2**k``. ``Fourier`` builds these
    on device for the encoder inputs and ``SpatialRotary`` builds them for the
    Q/K rotations; both call this so the two cannot disagree about what "the same
    frequency basis" means.
    """

    return tuple((2.0 * math.pi / period) * (2.0**k) for k in range(n_freqs))


def observation_contract(ship_config: Any) -> dict[str, Any]:
    """Small explicit descriptor of learned feature layout and semantics."""

    world_size = (
        ship_config["world_size"] if isinstance(ship_config, Mapping) else ship_config.world_size
    )
    return {
        "version": 10,
        "field_composition": "bounded_union_log_blend",
        "perception": "team_shared_range_field_core_los",
        "shot_reveal": "successful_fire_global_current_sample",
        "spawn_reveal": "visible_to_both_teams_for_the_spawn_decision",
        "hidden_tokens": "recursive_point_estimate_plus_age",
        # Every ship is seen on the decision it spawns, and validity is sticky,
        # so this is constant-true rather than a mask the trunk has to read.
        "belief_existence_mask": "always_valid_after_spawn",
        "auxiliary_prediction": "mean_plus_clamped_log_variance",
        "belief_uncertainty": "accumulated_forecast_variance_per_channel",
        "auxiliary_label_origin": "believed_current_to_true_next",
        "resource_targets": "normalised_scalar",
        "privileged_auxiliary_targets": "storage_only_never_policy_input",
        "enemy_actions": "always_private",
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

    v8 retains previously observed hidden enemies as recursively predicted point
    estimates, adds observation age and a non-privileged token-validity mask,
    and separates authoritative auxiliary targets from policy observations.
    v7 globally reveals a ship on the state sample where it successfully fires.
    v6 adds typed map tokens, independently masked team views, explicit
    visibility, private enemy actions, and the range/field-core LOS contract.
    v5 replaces parent-relative field channels with one absolute target
    log-index and changes overlap composition. v4 made the world-size-dependent
    base-2 position-frequency count explicit;
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
