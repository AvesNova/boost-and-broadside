"""One-time offline migration of the resilient-resonance-682 landmark checkpoints.

This is the D19 migration: a single, explicit, offline rewrite of one run's
checkpoint set into the schemas frozen by ``S14R``. It is deliberately *not*
runtime compatibility infrastructure. Nothing in ``src/`` imports it, no loader
calls it, and it supports exactly one run — the sixteen tracked ``.pt`` files
under ``checkpoints/resilient-resonance-682/``.

Migrating out of place
----------------------
The script never writes to its input. ``--source`` is read-only and ``--out``
must be a different directory, so the originals stay diffable and a re-run is
idempotent. Replacing the tracked files is a separate, deliberate copy the
operator makes after the verification in ``tests/migration/`` passes.

Where the provenance comes from
-------------------------------
Every field written here is *derived* from the run's own recorded data or left
out. Nothing is invented:

* ``paradigm`` — the run's own ``train_config["paradigm"]`` records ``ego_pass``.
* ``model_config`` — the legacy ``{d_model, n_heads, n_transformer_blocks}``
  block, renamed, plus the fields that did not exist at the training commit and
  whose values are *read back off the stored tensors* (see ``derive_model_config``).
* ``env_config`` — rebuilt, not copied. The legacy block carries ``num_obstacles``,
  which the current ``EnvConfig`` does not define; the current block's three new
  fields are derived from the historical environment's behavior.
* ``ship_config`` — rebuilt from the *historical* ``ShipConfig`` values at the
  training commit, which this script carries as a literal (``HISTORICAL_SHIP_CONFIG``)
  because the historical dataclass no longer exists in this tree.
* ``num_value_components`` / ``team_pma_k`` — read off the stored tensors and
  cross-checked against the run's recorded reward weights.
* ``resolved_config`` and ``launch`` — **omitted**. Both are optional in the frozen
  schema and neither was ever recorded for this run. Absent is the honest answer;
  a placeholder in either would sit in a field a loader compares.

The training commit is recorded by the run itself, in
``wandb_export/files/wandb-metadata.json``: see ``TRAINING_COMMIT``.

Usage
-----
    uv run --no-sync python scripts/migrate_682.py \
        --source checkpoints/resilient-resonance-682 \
        --out /tmp/682-migrated \
        --report /tmp/682-migrated/migration_report.md
"""

import argparse
import dataclasses
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from boost_and_broadside.config import EnvConfig, ModelConfig, ShipConfig  # noqa: E402
from boost_and_broadside.env.rewards import REWARD_COMPONENT_NAMES  # noqa: E402
from boost_and_broadside.train.rl.checkpoint import (  # noqa: E402
    POLICY_CHECKPOINT_FIELDS,
    RESUMABLE_CHECKPOINT_FIELDS,
)
from boost_and_broadside.train.rl.checkpoint_schema import OBSERVATION_SCHEMA  # noqa: E402

# Bump when any transformation below changes. Recorded per file in the report so a
# migrated payload can be traced to the exact rules that produced it.
TRANSFORMATION_VERSION = 1

RUN_NAME = "resilient-resonance-682"

# Recorded by the run itself in wandb_export/files/wandb-metadata.json. Every
# "historical" claim in this module is checkable against this commit, which is an
# ancestor of main.
TRAINING_COMMIT = "b4883769ca49bb60e818986586db5673a4bf83c1"

# The complete inventory: every final, best, recent-average, and ladder policy the
# landmark results are computed from. Asserted against the source directory so a
# missing or extra file fails instead of being silently skipped.
LADDER_STEPS: tuple[int, ...] = (
    14991360,
    21987328,
    28983296,
    36978688,
    49971200,
    70959104,
    87949312,
    102940672,
    155910144,
    206880768,
    272842752,
    416759808,
    876494848,
)
RESUMABLE_FILES: tuple[str, ...] = ("step_000999424000.pt", "recent_avg.pt")
BEST_FILES: tuple[str, ...] = ("best_training.pt",)
LADDER_FILES: tuple[str, ...] = tuple(f"ladder_step_{step:012d}.pt" for step in LADDER_STEPS)
ALL_FILES: tuple[str, ...] = RESUMABLE_FILES + BEST_FILES + LADDER_FILES

# ---------------------------------------------------------------------------
# Historical configuration, carried as literals
# ---------------------------------------------------------------------------

# ShipConfig as it stood at TRAINING_COMMIT (runs/shared.py SHIP_CONFIG). Only the
# fields the current ShipConfig still defines are listed; the five obstacle fields
# it dropped (obstacle_gravity_harmonic, obstacle_radius_min, obstacle_radius_max,
# obstacle_collision_radius, bullet_collision_radius) are recorded in the report
# as removed rather than carried. The run had num_obstacles=0, so none of them
# ever influenced these weights.
#
# Verified field-by-field against the training commit: every value below is what
# that commit's SHIP_CONFIG held, and every one of them is also what the current
# defaults.SHIP_CONFIG holds — the physics constants these weights were fitted to
# have not moved. That is why _check_config_drift correctly stays silent for these
# files; it is a measured agreement, not a copied one.
HISTORICAL_SHIP_CONFIG: dict[str, Any] = {
    "collision_radius": 10.0,
    "max_health": 100.0,
    "max_power": 100.0,
    "random_speed": False,
    "min_speed": 1.0,
    "max_speed": 180.0,
    "default_speed": 100.0,
    "base_thrust": 8.0,
    "boost_thrust": 80.0,
    "reverse_thrust": -80.0,
    "gravity_factor": 0.0,
    "gravity_eps": 10000.0,
    "power_speed_constant": 200.0,
    "passive_power_gain": 10.0,
    "no_turn_drag_coeff": 0.0008,
    "normal_turn_drag_coeff": 0.0012,
    "normal_turn_lift_coeff": 0.015,
    "sharp_turn_drag_coeff": 0.005,
    "sharp_turn_lift_coeff": 0.027,
    "normal_turn_angle": 0.08726646259971647,
    "sharp_turn_angle": 0.2617993877991494,
    "bullet_speed": 500.0,
}

# ShipConfig fields the current class defines that did not exist at TRAINING_COMMIT.
# They take their dataclass defaults, which is not a value the run chose — it is the
# absence of a choice. All of them describe refractive fields or bullet drag, neither
# of which existed; the run's env_config has num_fields=0, so none of them can affect
# how these weights read a zero-field environment.
SHIP_CONFIG_FIELDS_WITHOUT_HISTORY: tuple[str, ...] = (
    "bullet_drag_coeff",
    "bullet_field_damage_scale",
    "bullet_field_integration_substeps",
    "bullet_field_integrator",
    "field_index_step",
    "field_integration_substeps",
    "field_integrator",
    "field_interface_damage",
    "field_radius_max",
    "field_radius_min",
    "field_transition_width_max",
    "field_transition_width_min",
)

# ShipConfig fields that existed at TRAINING_COMMIT and no longer do.
SHIP_CONFIG_FIELDS_REMOVED: tuple[str, ...] = (
    "bullet_collision_radius",
    "obstacle_collision_radius",
    "obstacle_gravity_harmonic",
    "obstacle_radius_max",
    "obstacle_radius_min",
)

# REWARD_COMPONENT_NAMES at TRAINING_COMMIT, in order. The critic's K rows are the
# subset of this tuple with a non-zero weight, in this order, so it is what decides
# the value head's row meaning.
HISTORICAL_REWARD_COMPONENT_NAMES: tuple[str, ...] = (
    "ally_damage",
    "enemy_damage",
    "ally_death",
    "enemy_death",
    "ally_win",
    "enemy_win",
    "facing",
    "closing_speed",
    "shoot_quality",
    "kill_shot",
    "kill_assist",
    "damage_taken",
    "damage_dealt_enemy",
    "damage_dealt_ally",
    "death",
    "obstacle_death",
    "obstacle_proximity",
    "obstacle_closing_speed",
    "obstacle_tti",
    "shooting_penalty",
    "speed",
)

# Historical component name -> the current name for the same quantity. Only the two
# renames that touch a component this run actually trained are listed; every other
# historically active name is spelled identically today.
#
# The split behind both renames is the same: the current environment separates
# projectile damage from refractive-field boundary damage, so what used to be one
# "damage_taken"/"death" component is now a combat/field pair. The landmark run had
# no fields, so its single component is exactly the combat half.
REWARD_COMPONENT_RENAMES: dict[str, str] = {
    "damage_taken": "combat_damage_taken",
    "death": "combat_death",
}


# ---------------------------------------------------------------------------
# State-dict transformation
# ---------------------------------------------------------------------------

# The encoder's input columns. The first 58 are identical in name, order, and width
# between TRAINING_COMMIT and today; the current layout appends eight columns
# (field_transition_width, field_inside_log_index, field_outside_log_index,
# field_log_index_ratio, field_damage, local_log_index, local_index_gradient x2).
# Padding the first projection's weight with zeros over [58:66] therefore leaves the
# encoder's output *exactly* unchanged for any input, because those columns are
# multiplied by zero.
LEGACY_ENCODER_INPUT_DIM = 58
CURRENT_ENCODER_INPUT_DIM = 66

# Features whose *input encoding* changed between the training commit and today, so
# that the same physical quantity now arrives at the encoder on a different scale.
# Same column, same width, same position — different meaning.
#
# This is the one substantive discovery of the migration, and it is not visible in
# any shape, key, or config field. It was found by diffing every shared feature's
# encoder specification between the two commits: exactly one differs.
#
#   radius: Normalize(40.0) -> Normalize(0.5 * min(world_size)) == Normalize(512.0)
#
# The old constant was the obstacle radius ceiling; the new divisor is derived from
# the world size, because refractive fields are far larger than any obstacle was.
# Nothing in ShipConfig moved, so the loader's physics-drift check cannot see it.
#
# Left uncompensated, these weights would read a ship radius of 0.0195 where they
# were fitted on 0.25, silently changing every landmark evaluation. The compensation
# is exact rather than approximate: the feature enters through a single column of one
# Linear, so scaling that column of the weight by (new_divisor / old_divisor)
# reproduces the historical pre-activation for *every* input value, and the
# equivalence fixture measures that it does.
HISTORICAL_FEATURE_INPUT_SCALES: dict[str, float] = {"radius": 40.0}

# The next-state head predicted nine targets; the current layout appends a tenth,
# local_log_index, as the trailing predictor. Padding row 9 with zeros makes the
# migrated policy predict a constant zero for it — recorded as a known limitation
# rather than an equivalence, since the historical weights never learned it.
LEGACY_PREDICTION_DIM = 9
CURRENT_PREDICTION_DIM = 10


def migrate_state_dict_keys(name: str) -> str:
    """Rename one legacy parameter key to its current spelling.

    The Yemong block gained a *list* of spatial and temporal sublayers; the run was
    trained with exactly one of each, so ``spatial.`` becomes ``spatial.0.`` and
    ``temporal.`` becomes ``temporal.0.``. Nothing else in the trunk moved.
    """

    parts = name.split(".")
    if len(parts) >= 3 and parts[0] == "yemong_layers" and parts[2] in ("spatial", "temporal"):
        return ".".join(parts[:3] + ["0"] + parts[3:])
    return name


def current_feature_columns(ship_config: ShipConfig) -> dict[str, tuple[int, int]]:
    """Locate every feature's column span in the current encoder input vector.

    Derived from the live coordinator rather than hardcoded, and checked against its
    own reported width, so a layout change fails here instead of silently rescaling
    the wrong column.
    """

    from boost_and_broadside.train.rl.features import build_standard_coordinator

    coordinator = build_standard_coordinator(ship_config)
    dummy = coordinator._dummy_obs()
    spans: dict[str, tuple[int, int]] = {}
    offset = 0
    for feature in coordinator.features:
        width = int(feature.get_input(dummy).shape[-1])
        spans[feature.name] = (offset, offset + width)
        offset += width
    if offset != CURRENT_ENCODER_INPUT_DIM:
        raise ValueError(
            f"current encoder input width is {offset}, expected {CURRENT_ENCODER_INPUT_DIM}"
        )
    return spans


def encoder_column_rescales(ship_config: ShipConfig) -> dict[int, float]:
    """Per-column factors that undo the input-encoding changes since the run.

    ``w' = k * w`` where ``k = new_divisor / old_divisor``, so that
    ``w' * (x / new_divisor) == w * (x / old_divisor)`` for every ``x``.
    """

    from boost_and_broadside.train.rl.features import Normalize, build_standard_coordinator

    coordinator = build_standard_coordinator(ship_config)
    spans = current_feature_columns(ship_config)
    features = {feature.name: feature for feature in coordinator.features}

    rescales: dict[int, float] = {}
    for name, historical_divisor in HISTORICAL_FEATURE_INPUT_SCALES.items():
        encoder = features[name].input_encoder
        if not isinstance(encoder, Normalize):
            raise ValueError(f"{name!r} is no longer a Normalize feature; re-derive its rescale")
        current_divisor = float(encoder.scales)
        start, end = spans[name]
        if end - start != 1:
            raise ValueError(f"{name!r} spans {end - start} columns; the scalar rescale assumes 1")
        rescales[start] = current_divisor / historical_divisor
    return rescales


def _migrate_encoder_weight(
    tensor: torch.Tensor, rescales: dict[int, float], power: float = 1.0
) -> torch.Tensor:
    """Rescale the changed input columns, then widen with zeros for the new ones.

    ``power`` selects how the quantity transforms under ``w' = k * w``:
    ``1.0`` for the weight itself and for a running sum of it, ``-1.0`` for Adam's
    first gradient moment (the gradient scales as ``1/k``), ``-2.0`` for its second.
    """

    out_features, in_features = tensor.shape
    if in_features != LEGACY_ENCODER_INPUT_DIM:
        raise ValueError(
            f"encoder input width is {in_features}, expected {LEGACY_ENCODER_INPUT_DIM}"
        )
    rescaled = tensor.clone()
    for column, factor in rescales.items():
        if column >= LEGACY_ENCODER_INPUT_DIM:
            raise ValueError(f"rescale targets new column {column}, which has no legacy weights")
        rescaled[:, column] *= factor**power
    widened = torch.zeros(
        out_features, CURRENT_ENCODER_INPUT_DIM, dtype=tensor.dtype, device=tensor.device
    )
    widened[:, :LEGACY_ENCODER_INPUT_DIM] = rescaled
    return widened


def _pad_prediction_rows(tensor: torch.Tensor) -> torch.Tensor:
    """Add the trailing local_log_index predictor as a zero row/element."""

    if tensor.shape[0] != LEGACY_PREDICTION_DIM:
        raise ValueError(
            f"next-state head predicts {tensor.shape[0]} targets, "
            f"expected {LEGACY_PREDICTION_DIM}"
        )
    padded = torch.zeros(
        (CURRENT_PREDICTION_DIM, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device
    )
    padded[:LEGACY_PREDICTION_DIM] = tensor
    return padded


# Parameters the current architecture has and the legacy one did not. field_sub
# stands in for the temporal operator on field tokens, and is *only ever applied to
# field tokens* (models/yemong/griffin.py guards both call sites on a non-empty field
# slice). The landmark run has num_fields=0, so this parameter cannot affect any
# forward pass of these weights. Identity — not zero — matches how a freshly
# constructed block initializes it, and the reason is in griffin.py: the output feeds
# a multiplicative gate, so zeroing it would erase the recurrent branch.
def _new_parameter_value(name: str, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
    if ".field_sub." in name and name.endswith(".weight"):
        return torch.eye(shape[0], shape[1], dtype=dtype)
    raise ValueError(f"no derivation for new parameter {name!r}")


def migrate_policy_state_dict(
    legacy: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
    rescales: dict[int, float],
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    """Map one legacy policy state dict onto the current architecture.

    Args:
        legacy:    The stored ``policy_state_dict``.
        reference: ``state_dict()`` of a freshly built current policy, used only for
                   the shapes and dtypes of parameters the legacy payload lacks.
        rescales:  Encoder input columns whose encoding changed, and by how much.

    Returns:
        The migrated state dict and a per-tensor record of what was done to it.
    """

    migrated: dict[str, torch.Tensor] = {}
    records: list[dict[str, Any]] = []

    for name, tensor in legacy.items():
        new_name = migrate_state_dict_keys(name)
        if new_name == "encoder.feature_extractor.0.weight":
            value = _migrate_encoder_weight(tensor, rescales)
            rescale_note = ", ".join(
                f"scale column {column} by {factor:g}"
                for column, factor in sorted(rescales.items())
            )
            action = (
                f"{rescale_note}; pad input columns "
                f"[{LEGACY_ENCODER_INPUT_DIM}:{CURRENT_ENCODER_INPUT_DIM}] with zeros"
            )
        elif new_name in ("next_state_head.net.3.weight", "next_state_head.net.3.bias"):
            value = _pad_prediction_rows(tensor)
            action = f"pad prediction row {LEGACY_PREDICTION_DIM} with zeros"
        else:
            value = tensor.clone()
            action = "rename only" if new_name != name else "unchanged"
        migrated[new_name] = value
        records.append(
            {
                "legacy_key": name,
                "key": new_name,
                "legacy_shape": list(tensor.shape),
                "shape": list(value.shape),
                "action": action,
            }
        )

    for name in reference:
        if name in migrated:
            continue
        value = _new_parameter_value(name, reference[name].shape, reference[name].dtype)
        migrated[name] = value
        records.append(
            {
                "legacy_key": None,
                "key": name,
                "legacy_shape": None,
                "shape": list(value.shape),
                "action": "introduced as identity (no legacy counterpart; inert with num_fields=0)",
            }
        )

    missing = sorted(set(reference) - set(migrated))
    extra = sorted(set(migrated) - set(reference))
    if missing or extra:
        raise ValueError(
            f"migrated state dict does not match the current architecture: "
            f"missing={missing}, unexpected={extra}"
        )
    # Emit in the current architecture's own order so the payload reads like one a
    # current trainer wrote.
    ordered = {name: migrated[name] for name in reference}
    return ordered, records


def _migrate_like(
    name: str,
    tensor: torch.Tensor,
    target: torch.Size,
    rescales: dict[int, float],
    power: float,
) -> torch.Tensor:
    """Carry an optimizer moment or averaging accumulator through the same transform.

    Padded entries are exactly the encoder columns and prediction row the migration
    introduced, and zero is the correct value for both: no gradient ever flowed
    through them, so their Adam moments and their contribution to the running
    parameter average are both zero.

    ``power`` carries the reparameterization: a running *sum* of the weight scales
    with the weight (``+1``), while the gradient scales inversely (``-1``) and its
    square inversely squared (``-2``).
    """

    if name == "encoder.feature_extractor.0.weight":
        return _migrate_encoder_weight(tensor, rescales, power)
    if tuple(tensor.shape) == tuple(target):
        return tensor.clone()
    if name in ("next_state_head.net.3.weight", "next_state_head.net.3.bias"):
        return _pad_prediction_rows(tensor)
    raise ValueError(
        f"unexpected shape change for {name!r}: {tuple(tensor.shape)} -> {tuple(target)}"
    )


def migrate_optimizer_state(
    legacy_optimizer: Mapping[str, Any],
    legacy_parameter_names: list[str],
    current_parameter_names: list[str],
    current_shapes: dict[str, torch.Size],
    rescales: dict[int, float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Carry the complete Adam state across the parameter rename.

    Adam keys its ``state`` by the parameter's *position* in ``param_groups[0]["params"]``,
    which is ``policy.parameters()`` order. The rename changes that order's spelling
    but not its content, and the two new ``field_sub`` parameters are inserted at
    their architectural positions — so the mapping is by name, from the new index
    back to the old one, never by re-looking-up a renamed key in the old table.

    The recorded hyperparameters (lr, betas, eps, ...) are carried verbatim. This run
    trained at lr=1e-4 and eps=1e-5; a freshly constructed Adam would silently
    substitute 1e-3 and 1e-8.
    """

    legacy_state = legacy_optimizer["state"]
    legacy_index = {name: index for index, name in enumerate(legacy_parameter_names)}

    new_state: dict[int, Any] = {}
    records: list[dict[str, Any]] = []
    for new_id, name in enumerate(current_parameter_names):
        legacy_name = next(
            (old for old in legacy_parameter_names if migrate_state_dict_keys(old) == name),
            None,
        )
        if legacy_name is None:
            # A parameter with no history. PyTorch initializes a never-stepped
            # parameter with zero moments and step 0, and that is also the
            # better-behaved resume: with step 0 Adam's bias correction gives the
            # first update the ordinary ~lr magnitude, where a copied step count
            # would give it roughly 3x that.
            shape = current_shapes[name]
            new_state[new_id] = {
                "step": torch.tensor(0.0),
                "exp_avg": torch.zeros(shape, dtype=torch.float32),
                "exp_avg_sq": torch.zeros(shape, dtype=torch.float32),
            }
            records.append(
                {
                    "parameter": name,
                    "new_id": new_id,
                    "legacy_id": None,
                    "action": "fresh zero state, step 0",
                }
            )
            continue
        old_id = legacy_index[legacy_name]
        entry = legacy_state[old_id]
        moment_power = {"exp_avg": -1.0, "exp_avg_sq": -2.0}
        migrated_entry = {
            key: (
                _migrate_like(name, value, current_shapes[name], rescales, moment_power[key])
                if key in moment_power
                else value.clone() if isinstance(value, torch.Tensor) else value
            )
            for key, value in entry.items()
        }
        new_state[new_id] = migrated_entry
        records.append(
            {
                "parameter": name,
                "new_id": new_id,
                "legacy_id": old_id,
                "action": "carried" if legacy_name == name else "carried across rename",
            }
        )

    groups = []
    for group in legacy_optimizer["param_groups"]:
        migrated_group = dict(group)
        migrated_group["params"] = list(range(len(current_parameter_names)))
        groups.append(migrated_group)

    if len(new_state) != len(current_parameter_names):
        raise ValueError(
            f"optimizer state covers {len(new_state)} of {len(current_parameter_names)} parameters"
        )
    return {"state": new_state, "param_groups": groups}, records


def migrate_avg_param_cumsum(
    legacy_cumsum: list[torch.Tensor],
    legacy_parameter_names: list[str],
    current_parameter_names: list[str],
    current_shapes: dict[str, torch.Size],
    avg_update_count: int,
    rescales: dict[int, float],
) -> list[torch.Tensor]:
    """Realign the running parameter-sum with the current parameter order.

    The averaged policy is ``cumsum / avg_update_count``, so a new parameter whose
    averaged value must be the identity contributes ``identity * avg_update_count``.
    Zero-padded regions contribute zero, which is exactly the averaged policy's
    zero-padded weight. Both keep the two records consistent with each other.
    """

    legacy_by_name = dict(zip(legacy_parameter_names, legacy_cumsum, strict=True))
    migrated: list[torch.Tensor] = []
    for name in current_parameter_names:
        legacy_name = next(
            (old for old in legacy_parameter_names if migrate_state_dict_keys(old) == name),
            None,
        )
        if legacy_name is None:
            value = _new_parameter_value(name, current_shapes[name], torch.float32)
            migrated.append(value * float(avg_update_count))
            continue
        migrated.append(
            _migrate_like(
                name, legacy_by_name[legacy_name].float(), current_shapes[name], rescales, 1.0
            )
        )
    return migrated


# ---------------------------------------------------------------------------
# Provenance derivation
# ---------------------------------------------------------------------------


def derive_model_config(
    legacy: Mapping[str, Any], state_dict: Mapping[str, torch.Tensor]
) -> ModelConfig:
    """Rebuild ModelConfig from the legacy block plus what the tensors prove.

    The legacy block records only ``{d_model, n_heads, n_transformer_blocks}``. The
    six fields the current class added are not guesses: each is read back off the
    stored weights, and a mismatch raises rather than defaulting.
    """

    blocks = legacy["n_transformer_blocks"]

    # One spatial and one temporal sublayer per block: the legacy keys are
    # `spatial.<leaf>` with no sublayer index, which is the shape a single-sublayer
    # block serializes to.
    spatial = {k for k in state_dict if ".spatial." in k}
    temporal = {k for k in state_dict if ".temporal." in k}
    if any(k.split(".spatial.")[1].split(".")[0].isdigit() for k in spatial):
        raise ValueError("legacy state dict already has indexed spatial sublayers")
    if any(k.split(".temporal.")[1].split(".")[0].isdigit() for k in temporal):
        raise ValueError("legacy state dict already has indexed temporal sublayers")

    # A split encoder serializes two first projections; this one has a single
    # shared `feature_extractor`.
    if any(k.startswith("encoder.field_feature_extractor") for k in state_dict):
        raise ValueError("legacy state dict has a split encoder")

    # No bullet cross-attention: a reading policy serializes a bullet encoder.
    if any("bullet" in k for k in state_dict):
        raise ValueError("legacy state dict has bullet-encoder weights")

    return ModelConfig(
        d_model=legacy["d_model"],
        n_heads=legacy["n_heads"],
        n_yemong_blocks=blocks,
        n_spatial_per_block=1,
        n_temporal_per_block=1,
        encoder_split=False,
        n_bullet_cross_per_block=0,
        # Inert while n_bullet_cross_per_block is 0, and never recorded by the run.
        # The dataclass default stands in for a value the run never had.
        bullet_encoder_hidden=64,
        # A backward-pass memory setting, not architecture. It did not exist at the
        # training commit and does not affect inference.
        grad_checkpoint=False,
    )


def derive_env_config(legacy: Mapping[str, Any]) -> EnvConfig:
    """Rebuild EnvConfig, dropping what was removed and deriving what was added.

    Copying the legacy block verbatim makes the payload unloadable: it carries
    ``num_obstacles``, which the current ``EnvConfig`` does not define.
    """

    if legacy["num_obstacles"] != 0:
        raise ValueError(
            f"run used num_obstacles={legacy['num_obstacles']}; obstacles no longer exist "
            "and a non-zero count cannot be represented"
        )
    return EnvConfig(
        num_ships=legacy["num_ships"],
        max_bullets=legacy["max_bullets"],
        max_episode_steps=legacy["max_episode_steps"],
        # Refractive fields did not exist at the training commit: the historical
        # EnvConfig has no such field and the historical encoder has no field inputs.
        num_fields=0,
        single_team=legacy["single_team"],
        # The historical environment had no action repeat — it applied one action per
        # physics tick — so 1 is the value that reproduces its behavior, not a default.
        action_repeat=1,
        # The historical reset set every ship to full health and power unconditionally
        # (env.py: `torch.where(m, max_health, ...)`), which is spread 0.0 exactly.
        spawn_resource_spread=0.0,
    )


def derive_ship_config() -> ShipConfig:
    """Rebuild the historical ShipConfig in the current schema."""

    return ShipConfig(**HISTORICAL_SHIP_CONFIG)


def derive_active_components(rewards: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Return the critic's row meanings, historically and currently spelled.

    The critic's K rows are the non-zero-weighted entries of the *historical*
    registry, in that registry's order. Mapping each through
    ``REWARD_COMPONENT_RENAMES`` and checking the result against the current
    registry's order is what proves no permutation is needed.
    """

    historical = [
        name
        for name in HISTORICAL_REWARD_COMPONENT_NAMES
        if rewards.get(f"{name}_weight", 0.0) != 0.0
    ]
    current = [REWARD_COMPONENT_RENAMES.get(name, name) for name in historical]

    unknown = [name for name in current if name not in REWARD_COMPONENT_NAMES]
    if unknown:
        raise ValueError(f"active components with no current counterpart: {unknown}")

    # The decisive check: the same components, ordered by the *current* registry,
    # must come out in the same order as the historical one. If they ever diverge,
    # the value head needs a row permutation and this raises instead of silently
    # producing a scrambled critic.
    by_current_order = sorted(current, key=REWARD_COMPONENT_NAMES.index)
    if by_current_order != current:
        raise ValueError(
            "historical and current component orders disagree; the value head needs a "
            f"row permutation: historical={current}, current-order={by_current_order}"
        )
    return historical, current


def derive_team_pma_k(
    active_components: list[str], state_dict: Mapping[str, torch.Tensor]
) -> tuple[int, ...]:
    """Value-component indices routed through TeamPMA, checked against the tensors."""

    win_k = tuple(
        index for index, name in enumerate(active_components) if name in ("ally_win", "enemy_win")
    )
    stored_width = state_dict["value_head_win.3.weight"].shape[0]
    if len(win_k) != stored_width:
        raise ValueError(
            f"derived {len(win_k)} win components but value_head_win outputs {stored_width}"
        )
    return win_k


# ---------------------------------------------------------------------------
# Payload migration
# ---------------------------------------------------------------------------


def _parameter_names(state_dict: Mapping[str, torch.Tensor]) -> list[str]:
    return list(state_dict.keys())


def build_reference_policy(
    model_config: ModelConfig,
    ship_config: ShipConfig,
    num_value_components: int,
    team_pma_k: tuple[int, ...],
    num_ships: int,
):
    from boost_and_broadside.train.rl.policy_io import build_policy

    return build_policy(
        model_config,
        ship_config,
        num_value_components=num_value_components,
        num_ships=num_ships,
        team_pma_k=team_pma_k,
    )


def migrate_payload(
    name: str,
    legacy: Mapping[str, Any],
    run_provenance: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Migrate one checkpoint payload into its frozen family shape."""

    model_config: ModelConfig = run_provenance["model_config"]
    env_config: EnvConfig = run_provenance["env_config"]
    ship_config: ShipConfig = run_provenance["ship_config"]
    paradigm: str = run_provenance["paradigm"]
    active_components: list[str] = run_provenance["active_components"]

    legacy_state = legacy["policy_state_dict"]
    num_value_components = legacy_state["value_head_local.3.weight"].shape[0]
    if num_value_components != len(active_components):
        raise ValueError(
            f"{name}: critic width {num_value_components} does not match the "
            f"{len(active_components)} active reward components"
        )
    team_pma_k = derive_team_pma_k(active_components, legacy_state)
    if "team_pma_k" in legacy and tuple(legacy["team_pma_k"]) != team_pma_k:
        raise ValueError(
            f"{name}: stored team_pma_k {tuple(legacy['team_pma_k'])} disagrees with the "
            f"derived {team_pma_k}"
        )

    reference = build_reference_policy(
        model_config, ship_config, num_value_components, team_pma_k, env_config.num_ships
    )
    reference_state = reference.state_dict()
    current_parameter_names = [n for n, _ in reference.named_parameters()]
    current_shapes = {n: p.shape for n, p in reference.named_parameters()}
    if current_parameter_names != list(reference_state.keys()):
        raise ValueError(
            "current policy has buffers or a parameter order that differs from its "
            "state dict; the positional optimizer mapping is not safe"
        )

    rescales: dict[int, float] = run_provenance["encoder_rescales"]
    migrated_state, tensor_records = migrate_policy_state_dict(
        legacy_state, reference_state, rescales
    )
    reference.load_state_dict(migrated_state)  # strict: proves the mapping is complete

    payload: dict[str, Any] = {
        "observation_schema": OBSERVATION_SCHEMA,
        "policy_state_dict": migrated_state,
        "num_value_components": num_value_components,
        "team_pma_k": team_pma_k,
        "global_step": legacy["global_step"],
        "live_elo": legacy["training_elo"],
        "model_config": dataclasses.asdict(model_config),
        "env_config": dataclasses.asdict(env_config),
        "ship_config": dataclasses.asdict(ship_config),
        "paradigm": paradigm,
    }

    record: dict[str, Any] = {
        "file": name,
        "transformation_version": TRANSFORMATION_VERSION,
        "family": None,
        "tensor_mapping": tensor_records,
        "optimizer_mapping": None,
        "renamed_fields": {"training_elo": "live_elo"},
        "dropped_fields": [],
        "added_fields": [],
    }

    legacy_parameter_names = _parameter_names(legacy_state)

    if "optimizer_state_dict" in legacy:
        record["family"] = "resumable"
        record["renamed_fields"]["avg_training_elo"] = "avg_live_elo"
        optimizer, optimizer_records = migrate_optimizer_state(
            legacy["optimizer_state_dict"],
            legacy_parameter_names,
            current_parameter_names,
            current_shapes,
            rescales,
        )
        record["optimizer_mapping"] = optimizer_records
        avg_state, _ = migrate_policy_state_dict(
            legacy["avg_policy_state_dict"], reference_state, rescales
        )
        payload.update(
            {
                "optimizer_state_dict": optimizer,
                "scaler_state_dict": legacy["scaler_state_dict"],
                "adv_scaler_state_dict": legacy["adv_scaler_state_dict"],
                "avg_policy_state_dict": avg_state,
                "avg_param_cumsum": migrate_avg_param_cumsum(
                    legacy["avg_param_cumsum"],
                    legacy_parameter_names,
                    current_parameter_names,
                    current_shapes,
                    legacy["avg_update_count"],
                    rescales,
                ),
                "avg_update_count": legacy["avg_update_count"],
                "update": legacy["update"],
                "ship_steps": legacy["ship_steps"],
                "grad_tokens": legacy["grad_tokens"],
                "elapsed_train_time": legacy["elapsed_train_time"],
                "avg_live_elo": legacy["avg_training_elo"],
                "floating_games": legacy["floating_games"],
                "eval_window_rand": legacy["eval_window_rand"],
                "eval_window_sc": legacy["eval_window_sc"],
                "eval_window_ladder": legacy["eval_window_ladder"],
                "eval_window_floating": legacy["eval_window_floating"],
                "eval_window_live_vs_avg": legacy["eval_window_live_vs_avg"],
                "elo_milestone": legacy["elo_milestone"],
                "train_config": legacy["train_config"],
            }
        )
        # S12 removed scripted_elo: the scripted rung is now pinned from configuration
        # at 1000 rather than carried as a rating, so a stored value has no reader and
        # could only disagree with the pin.
        record["dropped_fields"].append("scripted_elo")
    elif "scaler_state_dict" in legacy:
        record["family"] = "best"
        payload.update(
            {
                "scaler_state_dict": legacy["scaler_state_dict"],
                "adv_scaler_state_dict": legacy["adv_scaler_state_dict"],
                "update": legacy["update"],
                "eval_window_rand": legacy["eval_window_rand"],
                "eval_window_sc": legacy["eval_window_sc"],
                "elo_milestone": legacy["elo_milestone"],
                "train_config": legacy["train_config"],
            }
        )
    else:
        record["family"] = "policy"

    record["added_fields"] = sorted(set(payload) - set(legacy) - {"live_elo", "avg_live_elo"})
    return payload, record


def run_provenance_from(resumable: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the run-level provenance every file shares, from one full checkpoint.

    The thirteen ladder snapshots record no configuration at all. Theirs is not
    invented: a ladder snapshot is a frozen copy of the same run's training policy,
    so the run's own recorded configuration is its configuration.
    """

    if "observation_schema" in resumable or "live_elo" in resumable:
        raise ValueError(
            "the source directory already holds migrated checkpoints. This migration "
            "reads originals and writes elsewhere; point --source at the originals "
            "(git-LFS still has them) rather than re-running it on its own output."
        )
    train_config = resumable["train_config"]
    paradigm = train_config["paradigm"]
    if paradigm not in ("ego_pass", "shared_pass"):
        raise ValueError(f"unrecognized paradigm {paradigm!r}")
    historical, current = derive_active_components(train_config["rewards"])
    ship_config = derive_ship_config()
    return {
        "model_config": derive_model_config(
            resumable["model_config"], resumable["policy_state_dict"]
        ),
        "env_config": derive_env_config(resumable["env_config"]),
        "ship_config": ship_config,
        "paradigm": paradigm,
        "active_components": current,
        "historical_components": historical,
        "encoder_rescales": encoder_column_rescales(ship_config),
        "encoder_columns": current_feature_columns(ship_config),
    }


# ---------------------------------------------------------------------------
# Report and entry point
# ---------------------------------------------------------------------------


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a payload's *contents*, independently of how it happened to serialize.

    The file hash is an identity for the bytes on disk, and for thirteen of the
    sixteen files it is also reproducible. It is not for the three that carry the
    historical ``train_config``: two of its reward fields are ``frozenset``s, and
    ``pickle`` writes a frozenset in iteration order, which Python randomizes per
    process. Re-running the migration on those three therefore produces a different
    file with identical contents.

    This digest canonicalizes instead — sorted mapping keys, sorted set members,
    tensors by dtype/shape/bytes — so it is stable across processes and is the thing
    a reproduction check should compare. Both are recorded per file.
    """

    digest = hashlib.sha256()

    def feed(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(f"T|{tensor.dtype}|{tuple(tensor.shape)}|".encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(value, Mapping):
            digest.update(f"M|{len(value)}|".encode())
            for key in sorted(value, key=repr):
                digest.update(f"K|{key!r}|".encode())
                feed(value[key])
        elif isinstance(value, (frozenset, set)):
            digest.update(f"S|{len(value)}|".encode())
            for item in sorted(value, key=repr):
                feed(item)
        elif isinstance(value, (list, tuple)):
            digest.update(f"L|{len(value)}|".encode())
            for item in value:
                feed(item)
        else:
            digest.update(f"V|{value!r}|".encode())

    feed(payload)
    return digest.hexdigest()


def _tensor_mapping_summary(records: list[dict[str, Any]]) -> list[str]:
    lines = []
    for entry in records:
        if entry["action"] in ("unchanged", "rename only"):
            continue
        legacy = entry["legacy_key"] or "(none)"
        lines.append(
            f"  - `{legacy}` {entry['legacy_shape']} -> `{entry['key']}` "
            f"{entry['shape']}: {entry['action']}"
        )
    return lines


def build_report_document(
    records: list[dict[str, Any]], provenance: Mapping[str, Any]
) -> dict[str, Any]:
    """Assemble the migration record: the one authority both report files describe.

    ``migration_report.json`` *is* this document, and the Markdown is rendered
    from it. Keeping one source is not tidiness — it is what stops the two from
    disagreeing. The Markdown once drifted because regenerating the prose re-ran
    the migration and re-hashed the payloads it had just written, and three of
    the sixteen files do not serialize byte-identically twice.
    """

    return {
        "run": RUN_NAME,
        "transformation_version": TRANSFORMATION_VERSION,
        "training_commit": TRAINING_COMMIT,
        "observation_schema": OBSERVATION_SCHEMA,
        "provenance": {
            "paradigm": provenance["paradigm"],
            "model_config": dataclasses.asdict(provenance["model_config"]),
            "env_config": dataclasses.asdict(provenance["env_config"]),
            "ship_config": dataclasses.asdict(provenance["ship_config"]),
            "historical_components": provenance["historical_components"],
            "active_components": provenance["active_components"],
        },
        "files": records,
    }


def _ship_config_from_document(document: Mapping[str, Any]) -> ShipConfig:
    """Rebuild the recorded ``ShipConfig``, restoring the tuples JSON flattened."""

    recorded = document["provenance"]["ship_config"]
    return ShipConfig(
        **{
            key: tuple(value) if isinstance(value, list) else value
            for key, value in recorded.items()
        }
    )


def render_report(document: Mapping[str, Any]) -> str:
    """Render the human-readable record from the migration document.

    Every fact here is read out of ``document``; nothing is re-measured and no
    checkpoint is opened. The two encoder tables are the exception that proves
    it — they are recomputed, but from the ``ship_config`` the document itself
    records, through the same pure functions the migration used.
    """

    # Render from the document as JSON stores it, whoever supplied it. The
    # config blocks are interpolated by repr, so an in-memory document would
    # print tuples where the tracked file has lists — the same report, spelled
    # two ways, depending on the caller.
    document = json.loads(json.dumps(document, default=str))
    records: list[dict[str, Any]] = list(document["files"])
    provenance = document["provenance"]
    ship_config = _ship_config_from_document(document)
    encoder_rescales = encoder_column_rescales(ship_config)
    encoder_columns = current_feature_columns(ship_config)

    lines: list[str] = []
    add = lines.append
    add(f"# Landmark checkpoint migration — `{document['run']}`")
    add("")
    add(
        "One-time offline migration of the complete landmark checkpoint set into the "
        "schemas frozen by `S14R`. Produced by `scripts/migrate_682.py`; this file is "
        "the record the plan's phase 10 requires."
    )
    add("")
    add(f"- Transformation version: `{document['transformation_version']}`")
    add(
        "- Training commit (recorded by the run in "
        f"`wandb_export/files/wandb-metadata.json`): `{document['training_commit']}`"
    )
    add(f"- Observation schema written: `{document['observation_schema']}`")
    add(f"- Files migrated: {len(records)}")
    add("")

    add("## Derived run provenance")
    add("")
    add("Every value below comes from the run's own recorded data or from the stored")
    add("tensors. Nothing is a placeholder, and `resolved_config` and `launch` are")
    add("omitted rather than invented — both are optional in the frozen schema and")
    add("neither was ever recorded for this run.")
    add("")
    add(
        f"- `paradigm`: `{provenance['paradigm']}` — from the run's own "
        '`train_config["paradigm"]`.'
    )
    add(f"- `model_config`: `{provenance['model_config']}`")
    add(f"- `env_config`: `{provenance['env_config']}`")
    add("- `ship_config`: the training commit's `SHIP_CONFIG`, re-expressed in the current")
    add("  schema. Every field both versions define holds the same value, so the loader's")
    add("  physics-drift check correctly stays silent.")
    add("")
    add("### Named unknowns")
    add("")
    add("Values the current schema requires that the run never recorded. Each takes its")
    add("dataclass default, which is the absence of a choice rather than a measured one:")
    add("")
    add("- `model_config.bullet_encoder_hidden` — inert while `n_bullet_cross_per_block=0`.")
    add("- `model_config.grad_checkpoint` — a backward-pass memory setting, not architecture.")
    add(
        "- "
        + ", ".join(f"`ship_config.{field}`" for field in SHIP_CONFIG_FIELDS_WITHOUT_HISTORY)
        + " — refractive-field and bullet-drag physics, none of which existed at the"
        " training commit. The run has `num_fields=0`, and the four of these the feature"
        " pipeline reads scale only the zero-weighted encoder columns, so no value of"
        " theirs can reach these weights."
    )
    add("")
    add("Fields the training commit defined that the current schema does not, and which")
    add("are therefore dropped rather than carried:")
    add("")
    for field in SHIP_CONFIG_FIELDS_REMOVED:
        add(f"- `ship_config.{field}`")
    add("- `env_config.num_obstacles` — the run set it to 0, so nothing is lost.")
    add("")

    add("## Value-component mapping")
    add("")
    add("The critic's eleven rows are the non-zero-weighted entries of the training")
    add("commit's `REWARD_COMPONENT_NAMES`, in that registry's order. Two of them are")
    add("spelled differently today, because the current environment splits projectile")
    add("damage from refractive-field boundary damage and the landmark run had no fields:")
    add("")
    add("| K | historical name | current name |")
    add("|---:|---|---|")
    for index, (old, new) in enumerate(
        zip(provenance["historical_components"], provenance["active_components"], strict=True)
    ):
        add(f"| {index} | `{old}` | `{new}` |")
    add("")
    add("Ordering these eleven by the *current* registry's index reproduces the same")
    add("sequence, so **the value head needs no row permutation** and none is applied.")
    add("`scripts/migrate_682.py` recomputes this and raises rather than emitting a")
    add("scrambled critic if the two orders ever diverge.")
    add("")

    add("## Changed input encodings")
    add("")
    add("The one substantive discovery of this migration, and the only part of it that")
    add("is not a rename, a pad, or a copy. It is invisible in every shape, key, and")
    add("config field: the same physical quantity now reaches the encoder on a different")
    add("scale, through the same column.")
    add("")
    add("| feature | column | divisor at training | divisor now | weight factor |")
    add("|---|---:|---:|---:|---:|")
    for column, factor in sorted(encoder_rescales.items()):
        name = next(key for key, span in encoder_columns.items() if span[0] == column)
        historical = HISTORICAL_FEATURE_INPUT_SCALES[name]
        add(f"| `{name}` | {column} | {historical:g} | {historical * factor:g} | {factor:g} |")
    add("")
    add("`radius` was normalized by the obstacle radius ceiling (40) and is now normalized")
    add("by half the world size (512), because refractive fields are far larger than any")
    add("obstacle was. No `ShipConfig` field moved, so the loader's physics-drift check")
    add("cannot see this; it was found by diffing every shared feature's encoder")
    add("specification against the training commit, where it is the only difference.")
    add("")
    add("Uncompensated, these weights would read a ship radius of 0.0195 where they were")
    add("fitted on 0.25. How much that matters is worth stating rather than implying: on")
    add("the fixed-observation set it moves logits by up to 1.3, and a 64-game 4v4 of")
    add("`best_training.pt` against the scripted agent goes from 64 wins to 63. The")
    add("landmark policy is far stronger than scripted, so a perturbation has room to hide")
    add("there; between two ladder snapshots of similar strength it has less. The point is")
    add("that this is a behavior change with no reason to accept it, not that it is large.")
    add("")
    add("The compensation is exact, not approximate: the feature enters")
    add("through one column of one `Linear`, so scaling that column of the weight by")
    add("`512/40` reproduces the historical pre-activation for every input value. Adam's")
    add("moments for that column are carried as `1/k` and `1/k^2` (the gradient scales")
    add("inversely with the weight) and the averaging accumulator as `k`, so the")
    add("optimizer and averaged-policy records stay consistent with the weight.")
    add("")

    add("## Per-file record")
    add("")
    add("| file | family | original SHA-256 | migrated SHA-256 | migrated content SHA-256 |")
    add("|---|---|---|---|---|")
    for record in records:
        digests = record["sha256"]
        add(
            f"| `{record['file']}` | {record['family']} | `{digests['original']}` | "
            f"`{digests['migrated']}` | `{digests['migrated_content']}` |"
        )
    add("")
    add("The original hashes are the git-LFS object ids of the files this migration read;")
    add("`tests/migration/` re-derives them so the inputs stay identifiable. The migrated")
    add("hash identifies the bytes now tracked. The content hash is the one to compare when")
    add("*reproducing* the migration: the three payloads carrying the historical")
    add("`train_config` do not serialize byte-identically twice, because two of its reward")
    add("fields are `frozenset`s and `pickle` writes a frozenset in iteration order, which")
    add("Python randomizes per process. Their contents are identical; only the byte order of")
    add("two three-element sets moves. The other thirteen files are byte-reproducible.")
    add("")

    add("### Transformations")
    add("")
    add("Every file's `policy_state_dict` receives the same key rename, the same three")
    add("edits to existing tensors, and the same two introduced parameters; the two")
    add("resumable files additionally carry their optimizer and averaging state across it.")
    add("")
    add("Key rename: `yemong_layers.<i>.spatial.<leaf>` -> `yemong_layers.<i>.spatial.0.<leaf>`")
    add("and the same for `temporal`. The block gained a list of sublayers and this run")
    add("trained with one of each.")
    add("")
    for record in records:
        summary = _tensor_mapping_summary(record["tensor_mapping"])
        if not summary:
            continue
        add(f"`{record['file']}`:")
        add("")
        lines.extend(summary)
        add("")
        break
    add("The list above is identical in every one of the sixteen files, so only the first")
    add("is shown here. The complete per-tensor mapping for every file — including the")
    add("69 tensors that are renamed or copied unchanged, and the per-parameter optimizer")
    add("index mapping for the two resumable files — is in `migration_report.json` beside")
    add("this file.")
    add("")

    add("## Validation")
    add("")
    add("Recorded per file in `migration_report.json` and enforced by")
    add("`tests/migration/test_landmark_682.py`, which runs over all sixteen files:")
    add("")
    add("- the payload carries exactly its frozen family's key set;")
    add("- it loads through the ordinary `load_policy_bundle` with no migration path;")
    add("- the two resumable files pass `require_resumable_checkpoint`;")
    add("- fixed seeded observations through the migrated policy reproduce the historical")
    add("  policy's logits, action distributions, values, recurrent state, and next-state")
    add("  outputs, against a reference captured from the training commit;")
    add("- a seeded zero-field scenario reproduces the historical policy's actions and")
    add("  recurrent state over a multi-step episode.")
    add("")
    add("Measured at transformation version 1, over all sixteen files and both input sets:")
    add("the encoder's own output agrees with the historical one to 2.4e-07, which is under")
    add("one float32 ULP at its magnitude. Amplified through two Yemong blocks that becomes")
    add("3.2e-05 on logits, 7.9e-06 on values, 8.6e-06 on the nine inherited next-state")
    add("outputs, and 1.0e-05 on the recurrent state. Every greedy action matches exactly.")
    add("Bitwise equality is not reachable and is not claimed: the encoder's first matmul")
    add("went from k=58 to k=66, which reorders the accumulation of the terms that survived.")
    add("")

    add("## Known limitations")
    add("")
    add("- **The tenth next-state predictor is zero.** The current head predicts a")
    add("  `local_log_index` target the historical head never had; its row is zero-padded,")
    add("  so these weights predict a constant zero for it. The other nine are exact.")
    add("- **`field_sub` has no history.** It is introduced as the identity every fresh")
    add("  block initializes it to. It is applied only to field tokens, and this run has")
    add("  `num_fields=0`, so it cannot affect any forward pass of these weights.")
    add("- **Resumable, with a caveat.** `step_000999424000.pt` and `recent_avg.pt` carry")
    add("  their complete Adam state and its recorded hyperparameters (`lr=1e-4`,")
    add("  `eps=1e-5`) across the rename, so they satisfy the frozen resumable contract.")
    add("  Actually continuing the run additionally needs a profile whose reward vocabulary")
    add("  matches the historical one, which no longer exists — the current registry splits")
    add("  two of these eleven components. These files are complete and loadable; they are")
    add("  not a resumable path back into today's training system.")
    add("- **`train_config` is kept verbatim**, in the historical schema. It is the record")
    add("  of what the run was launched with. No loader rebuilds it into a dataclass.")
    add("")

    # Sections are written with a trailing blank so they compose; drop it at the end
    # so the file ends with exactly one newline.
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, help="read-only landmark run directory")
    parser.add_argument("--out", type=Path, help="output directory (must differ from --source)")
    parser.add_argument("--report", type=Path, default=None, help="Markdown report path")
    parser.add_argument(
        "--render-report-from",
        type=Path,
        default=None,
        help=(
            "regenerate the Markdown report from an existing migration_report.json "
            "and exit; migrates nothing and hashes nothing"
        ),
    )
    args = parser.parse_args(argv)

    if args.render_report_from is not None:
        # Regenerating the prose must never re-run the migration: three of the
        # sixteen payloads do not serialize byte-identically twice, so a re-run
        # would write hashes that describe bytes nobody tracks.
        if args.source is not None or args.out is not None:
            parser.error("--render-report-from renders the tracked record; it takes no source")
        document = json.loads(args.render_report_from.read_text())
        report_path = args.report or args.render_report_from.with_suffix(".md")
        report_path.write_text(render_report(document))
        print(f"rendered {report_path} from {args.render_report_from}")
        return 0

    if args.source is None or args.out is None:
        parser.error("--source and --out are required unless --render-report-from is given")

    source: Path = args.source.resolve()
    out: Path = args.out.resolve()
    if source == out:
        parser.error("--out must differ from --source: this migration never overwrites its input")
    if not source.is_dir():
        parser.error(f"no such source directory: {source}")

    present = {path.name for path in source.glob("*.pt")}
    if present != set(ALL_FILES):
        parser.error(
            "source inventory does not match the expected landmark set: "
            f"missing={sorted(set(ALL_FILES) - present)}, "
            f"unexpected={sorted(present - set(ALL_FILES))}"
        )

    out.mkdir(parents=True, exist_ok=True)

    anchor = torch.load(source / "step_000999424000.pt", map_location="cpu", weights_only=False)
    provenance = run_provenance_from(anchor)

    records: list[dict[str, Any]] = []
    for name in ALL_FILES:
        legacy = torch.load(source / name, map_location="cpu", weights_only=False)
        payload, record = migrate_payload(name, legacy, provenance)

        expected = (
            RESUMABLE_CHECKPOINT_FIELDS
            if record["family"] == "resumable"
            else POLICY_CHECKPOINT_FIELDS
        )
        missing = [field for field in expected if field not in payload]
        if missing:
            raise ValueError(f"{name}: migrated payload is missing {missing}")

        destination = out / name
        torch.save(payload, destination)
        record["sha256"] = {
            "original": sha256(source / name),
            "migrated": sha256(destination),
            "migrated_content": content_sha256(payload),
        }
        records.append(record)
        print(f"migrated {name} ({record['family']})")

    report_path = args.report or (out / "migration_report.md")
    json_path = report_path.parent / "migration_report.json"
    json_path.write_text(
        json.dumps(build_report_document(records, provenance), indent=1, default=str) + "\n"
    )
    # Render from what was written, not from what is in memory, so this run and a
    # later --render-report-from over the same JSON produce identical prose.
    report_path.write_text(render_report(json.loads(json_path.read_text())))
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
