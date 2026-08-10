"""The BC profile's named, reviewed relationship to RL (plan §3, S11).

BC is rebuilt independently rather than derived from RL, so nothing stops the
two drifting apart silently again.  These tests are the replacement for
inheritance: every way resolved BC differs from resolved RL has to be listed
here with a reason, and the pre-correction S01 evidence pins what the correction
actually changed.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pytest

from boost_and_broadside.config.fingerprint import canonical_data
from boost_and_broadside.config.resolve import resolve_profile
from boost_and_broadside.config.schedule_spec import TrainingScheduleSpec
from boost_and_broadside.profiles import PROFILES

_ROOT = Path(__file__).resolve().parents[2]
_SNAPSHOTS = _ROOT / "tests" / "fixtures" / "mode_refactor"

# Resolved training values BC is allowed to differ from RL on, and why.  A new
# entry here is a deliberate divergence someone has to justify in review.
ALLOWED_TRAIN_CONFIG_DIFFERENCES = {
    "next_state_coef": (
        "BC weights next-state prediction at full strength while it has a dense "
        "supervised signal to learn the trunk from; RL runs it as a 0.2 auxiliary."
    ),
    "total_timesteps": (
        "BC owns its budget and stops when imitation saturates, not when RL's "
        "curriculum ends."
    ),
}

# Schedule entries BC is allowed to differ from RL on, and why.  Compared as
# whole declarative expressions: a differing shape is a differing entry.
ALLOWED_SCHEDULE_DIFFERENCES = {
    "learning_rate": (
        "Same warmup to the same project learning rate, then hold.  RL's decay "
        "tail is keyed to keypoints at the end of its own 500M-step budget."
    ),
    "policy_gradient_coef": "Disabled: BC takes no policy gradient.",
    "behavior_cloning_coef": (
        "In BC this is the policy head's only learning signal, balanced 1:1 "
        "against the next-state auxiliary; RL's 2.0 is an auxiliary term "
        "carried alongside a live policy gradient."
    ),
    "league_fraction": "Disabled: no roster opponent plays a BC rollout.",
    "target_kl": (
        "A KL trust region early-stops epochs when the policy leaves the one "
        "that produced the rollout; under supervision that movement is the "
        "objective, so the PPO stopping criterion does not apply."
    ),
}


_COLLAPSED_UNITS = ("component_gammas", "component_lambdas")


def _differing_train_config_paths(left: dict, right: dict, prefix: str = "") -> set[str]:
    """Return differing leaf paths, collapsed to reviewable units.

    A schedule entry and a per-component discount table each read as one
    decision, so ``schedule.learning_rate.closure.steps.0`` and
    ``component_gammas.ally_win`` report as ``schedule.learning_rate`` and
    ``component_gammas``.
    """

    if prefix.startswith("schedule."):
        name = prefix.split(".")[1]
        return {f"schedule.{name}"} if left != right else set()
    if prefix.startswith(_COLLAPSED_UNITS):
        return {prefix.split(".")[0]} if left != right else set()
    if isinstance(left, dict) and isinstance(right, dict):
        paths: set[str] = set()
        for key in left.keys() | right.keys():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.add(path)
            else:
                paths.update(_differing_train_config_paths(left[key], right[key], path))
        return paths
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        paths = set()
        for index, (one, other) in enumerate(zip(left, right, strict=True)):
            paths.update(_differing_train_config_paths(one, other, f"{prefix}.{index}"))
        return paths
    return {prefix} if left != right else set()


def test_resolved_bc_differs_from_rl_only_where_the_objective_requires_it() -> None:
    rl = canonical_data(resolve_profile(PROFILES["rl"]).train_config)
    bc = canonical_data(resolve_profile(PROFILES["bc"]).train_config)

    expected = set(ALLOWED_TRAIN_CONFIG_DIFFERENCES) | {
        f"schedule.{name}" for name in ALLOWED_SCHEDULE_DIFFERENCES
    }
    assert _differing_train_config_paths(rl, bc) == expected


def test_declarative_bc_schedule_differs_from_rl_only_on_named_entries() -> None:
    """Compare intent, not just the compiled closures the resolver produces."""

    rl = PROFILES["rl"].objective.schedule
    bc = PROFILES["bc"].objective.schedule
    differing = {
        field.name
        for field in fields(TrainingScheduleSpec)
        if getattr(rl, field.name) != getattr(bc, field.name)
    }
    assert differing == set(ALLOWED_SCHEDULE_DIFFERENCES)


def test_bc_shares_rls_environment_batch_and_optimizer_shape() -> None:
    """The values the correction pulled back in line, asserted directly."""

    rl = resolve_profile(PROFILES["rl"])
    bc = resolve_profile(PROFILES["bc"])
    rl_scale, bc_scale = rl.train_config.scales[0], bc.train_config.scales[0]

    assert canonical_data(bc_scale.env_config) == canonical_data(rl_scale.env_config)
    assert bc_scale.env_config.num_ships == 8
    assert bc_scale.env_config.action_repeat == 2
    assert bc_scale.num_envs == rl_scale.num_envs == 3904
    assert bc.train_config.num_steps == rl.train_config.num_steps
    assert bc.train_config.num_minibatches == rl.train_config.num_minibatches == 32
    assert bc.train_config.rollouts_per_update == rl.train_config.rollouts_per_update == 3
    assert bc.train_config.microbatch_tokens == rl.train_config.microbatch_tokens == 25_000
    assert bc.train_config.return_quantile_samples == 262_144
    assert bc.train_config.clip_coef == rl.train_config.clip_coef == 0.15
    assert (bc.train_config.gamma, bc.train_config.gae_lambda) == (0.9801, 0.9025)
    assert dict(bc.train_config.component_gammas) == dict(rl.train_config.component_gammas)
    assert dict(bc.train_config.component_lambdas) == dict(rl.train_config.component_lambdas)
    assert bc.train_config.reference_ladder == rl.train_config.reference_ladder
    assert bc.train_config.random_elo == rl.train_config.random_elo

    def batch_tokens(resolved) -> int:
        scale = resolved.train_config.scales[0]
        return (
            scale.num_envs
            * resolved.train_config.num_steps
            * (scale.env_config.num_ships + scale.env_config.num_fields)
            * resolved.train_config.rollouts_per_update
        )

    assert batch_tokens(bc) == batch_tokens(rl) == 11_993_088


@pytest.mark.parametrize("step", (0, 1, 6_000_000, 1_000_000_000, 2_000_000_000))
def test_bc_objective_holds_at_every_point_of_its_budget(step: int) -> None:
    """Disabling the policy gradient and the league is not a phase; it is the profile."""

    schedule = resolve_profile(PROFILES["bc"]).train_config.schedule
    assert schedule.policy_gradient_coef(step) == 0.0
    assert schedule.league_fraction(step) == 0.0
    assert schedule.behavior_cloning_coef(step) > 0.0
    assert schedule.target_kl(step) is None
    assert schedule.learning_rate(step) > 0.0


def test_bc_learning_rate_warms_up_and_then_holds() -> None:
    schedule = resolve_profile(PROFILES["bc"]).train_config.schedule
    assert schedule.learning_rate(0) == pytest.approx(1e-7)
    assert schedule.learning_rate(6_000_000) == pytest.approx(3e-4)
    assert schedule.learning_rate(2_000_000_000) == pytest.approx(3e-4)


def test_corrected_bc_changed_exactly_the_reviewed_stale_values() -> None:
    """Pin the S01 pre-correction evidence against the corrected profile.

    ``bc-stale.json`` is the resolved BC configuration as it stood before this
    section.  Anything that reverts toward it, or any further silent change,
    breaks here rather than in a training run.
    """

    stale = json.loads((_SNAPSHOTS / "bc-stale.json").read_text())["train_config"]
    corrected = canonical_data(resolve_profile(PROFILES["bc"]).train_config)

    assert _differing_train_config_paths(stale, corrected) == {
        # 4v4 in RL's environment, at RL's decision rate and spawn spread.
        "scales.0.env_config.num_ships",
        "scales.0.env_config.action_repeat",
        "scales.0.env_config.spawn_resource_spread",
        # Discounts follow the action repeat and gain the component tables.
        "gamma",
        "gae_lambda",
        "component_gammas",
        "component_lambdas",
        # RL's logical batch, token sizing, minibatching, microbatching, and
        # bounded return quantiles.
        "scales.0.num_envs",
        "rollouts_per_update",
        "num_minibatches",
        "microbatch_tokens",
        "return_quantile_samples",
        # Current project optimizer value.
        "clip_coef",
        # The live gauge BC's Elo evaluation and RL both rate on.
        "reference_ladder",
        "random_elo",
        # Schedule values RL moved and BC had not followed.
        "schedule.entropy_coef",
        "schedule.checkpoint_interval",
        # Representation-only: stepped((0, 4)) compiles to the same constant 4
        # RL uses, so the invariant above sees no num_epochs difference.
        "schedule.num_epochs",
    }


def test_corrected_bc_matches_its_reviewed_snapshot() -> None:
    expected = json.loads((_SNAPSHOTS / "bc.json").read_text())
    resolved = resolve_profile(PROFILES["bc"])

    assert canonical_data(resolved.ship_config) == expected["ship_config"]
    assert canonical_data(resolved.model_config) == expected["model_config"]
    assert canonical_data(resolved.train_config) == expected["train_config"]
