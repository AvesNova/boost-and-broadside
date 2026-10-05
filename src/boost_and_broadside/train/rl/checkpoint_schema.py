"""Checkpoint compatibility for the explicit policy-observation contract."""

import math
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from boost_and_broadside.train.rl.ship_codes import SHIP_CODE_DIM

OBSERVATION_SCHEMA = "categorical_codes_v21"
POSITION_FINEST_PERIOD = 128.0
# Harmonics of the attitude axis of rotary spatial attention. Defined here,
# beside the position count, because the rotation and the bullets' Fourier
# position feature share these bases.
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
        "version": 21,
        # Every mode presents one global/game token directly after the ships,
        # carrying a categorical game-mode one-hot, the match clock and (in
        # Frontline) the front. Whether the policy promotes it to a recurrent
        # query is ``ModelConfig.global_token``.
        "global_token": "permanent_after_ships_categorical_game_mode",
        "field_composition": "bounded_union_log_blend",
        "perception": "team_shared_range_field_core_los",
        "shot_reveal": "successful_fire_global_current_sample",
        "spawn_reveal": "visible_to_both_teams_for_the_spawn_decision",
        "hidden_tokens": "recursive_physical_belief_plus_age",
        # There is no substitution any more. The belief stores the same eleven
        # physical quantities truth does, so composing a legal view selects
        # between two tensors of one meaning and the ordinary encoder reads the
        # result. The spatial rotation reads the composed coordinate for the same
        # reason.
        "belief_substitution": "none_legal_source_selection_before_encoding",
        # Every ship is seen on the decision it spawns, and validity is sticky,
        # so this is constant-true rather than a mask the trunk has to read.
        # Not a mask any more: constant-true after spawn, so attention carries no
        # key padding and SDPA can reach the fused flash kernel. The channel
        # survives only to weight team pooling in the value head, and its
        # constant encoder feature is gone.
        "belief_existence_mask": "removed_constant_true_after_spawn",
        # Every predicted ship channel is read and predicted as one categorical
        # code (train/rl/ship_codes.py): 9-colour nested position (81), three
        # velocity axes (243), 4-colour attitude (16), angular velocity (41),
        # four bounded scalars (21 each) and cooldown (4).
        "ship_state_input": "categorical_code_469_rebuilt_from_belief_moments",
        "auxiliary_prediction": "residual_logits_on_input_code_per_group_cross_entropy",
        "next_state_targets": "exact_code_of_true_next_state",
        # Fourteen physical spreads, zero when certain: position sigma, raw
        # velocity covariance, five sigmas and the cooldown residual.
        "belief_uncertainty": "fourteen_physical_spreads_zero_is_certain",
        "ship_code_dim": SHIP_CODE_DIM,
        "privileged_auxiliary_targets": "storage_only_never_policy_input",
        "pending_action_features": "joint_42_probability_vector",
        "policy_action_distribution": "joint_categorical_3x7x2",
        "enemy_actions": "dedicated_prediction_with_spawn_null_override",
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

    v21 is the learning redesign's one bump (``frontline-redesign-plan.md`` Part
    II). Ship state enters the encoder as categorical codes instead of Fourier
    and symlog features, the next-state head predicts those codes as residual
    logits instead of Gaussian deltas, the belief's uncertainty becomes fourteen
    physical spreads, the reward has five levels with a categorical critic per
    level, and the outcome is valued as four classes off the global token.
    Every learned input and head width changes.

    v20 makes the global/game token permanent. It moves from the end of the
    token axis to directly after the ships, exists in combat as well as
    Frontline, and its ``game_mode`` channel widens from one scalar to a
    categorical one-hot. The encoder input width changes and a recurrent global
    token changes the hidden-state width, so no v19 checkpoint can load.

    v19 replaces the encoded belief with a physical one. Hidden ships now reach
    the trunk as ordinary physical channels selected before encoding rather than
    as Fourier moments substituted after it, the next-state head narrows from
    means-plus-spreads over the encoded target space to eleven physical deltas
    and thirteen uncertainty terms, and ``belief_uncertainty`` narrows to those
    thirteen. Both the encoder input width and the auxiliary head width change,
    and a v18 head's outputs mean something else entirely -- absolute encoded
    targets rather than physical deltas -- so no tensor-only migration exists
    even where a shape happens to match.

    v18 replaces the factorized pending-action input and private categories with
    a 42-way joint probability vector. It also adds a dedicated 42-way enemy
    action head, so both the encoder and model state shapes are incompatible.

    v17 replaces the three independent actor categoricals with one 42-command
    joint categorical and widens pending-action inputs from 3+7+2 to 4+8+3 so
    private opponent commands have explicit unknown categories. Both learned
    input and output projections change shape; older weights cannot load.
    v15 makes map objects key/value-only inputs to a single non-square
    attention: ship queries read N+M keys in one softmax instead of two summed
    ones. The trunk carries ship tokens only, so the block's parameter set and
    the map projection change; a v14 checkpoint cannot load.
    v14 turns on the per-entity-type first projection. The encoder's parameters
    change shape -- four typed projections plus a shared second layer, instead of
    one wide extractor -- so a v13 checkpoint cannot load.
    v13 removes the attention key-padding mask and the constant ``belief_valid``
    encoder feature. Measured at 0 of 33,280 false over a Frontline rollout, that
    mask encoded nothing while disqualifying the fused SDPA kernel. Encoder input
    narrows 109 -> 108.
    v12 drops the per-harmonic spread from the circular channels, keeping one on
    the finest harmonic of each. The auxiliary head narrows from 88 outputs to
    67 and the encoder input from 130 to 109, so a v11 checkpoint cannot load.
    v11 predicts every channel absolutely: velocity and the ship-local log index
    were the last two deltas. The tensor shapes are unchanged, which is exactly
    why this has to be gated -- a v10 head's velocity output means a *step* and a
    v11 head's means the state, so the weights would load cleanly and the belief
    recursion would then integrate a value that was never a step.
    v10 predicts position and attitude as absolute Fourier moments on the same
    harmonic basis their inputs use, with one isotropic spread per (sin, cos)
    pair, and makes health, power and cooldown normalised scalars on both the
    input and the target side. Both the encoder input width and the auxiliary
    head width change, and ``belief_uncertainty`` becomes one channel per
    uncertainty column -- which now scales with the world, because position
    reports a spread per harmonic. No tensor-only migration exists.
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
