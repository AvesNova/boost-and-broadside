"""Contracts for the profile-intent and resolved-configuration boundary.

Three source-scanning tests were removed from this file: an ``ast`` check that
the base profile imports no overlay, an ``ast`` check that ``config/`` imports no
runtime engine module, and a regex sweep for references to the deleted ``runs/``
package. Each stated a real rule by pattern-matching over source text, which
fires on renames and misses anything phrased differently. The rules hold; the
enforcement cost more than it caught.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields, replace
from pathlib import Path

import pytest

from boost_and_broadside.config.fingerprint import canonical_data, canonical_json, fingerprint
from boost_and_broadside.config.resolve import (
    _PASS_THROUGH,
    LaunchOverrides,
    derive_aligned_num_envs,
    derive_time_normalized_value,
    resolve_profile,
)
from boost_and_broadside.config.schedule_spec import compile_keypoints, hold
from boost_and_broadside.config.schema import LaunchSizingSpec, ProfileSpec
from boost_and_broadside.config.service import format_resolved_config, resolved_profile_document
from boost_and_broadside.config.training import TrainConfig
from boost_and_broadside.profiles import PROFILES

_ROOT = Path(__file__).resolve().parents[2]


def test_bc_overlays_rl_on_exactly_the_named_objective_differences() -> None:
    """BC's whole divergence from RL, as data.

    This replaces ``tests/config/test_bc_profile.py``, which spent 181 lines
    checking by hand that BC had not drifted on any *shared* value. An overlay
    cannot: a shared value that moves here moves in RL too. What is still worth
    pinning is the other direction -- that the list of deliberate differences is
    the one that was reviewed, and has not quietly grown.
    """

    rl = canonical_data(resolve_profile(PROFILES["rl"]).train_config)
    bc = canonical_data(resolve_profile(PROFILES["bc"]).train_config)

    def different_paths(left, right, prefix=""):
        if isinstance(left, dict) and isinstance(right, dict):
            paths = set()
            for key in left.keys() | right.keys():
                path = f"{prefix}.{key}" if prefix else key
                if key not in left or key not in right:
                    paths.add(path)
                else:
                    paths.update(different_paths(left[key], right[key], path))
            return paths
        if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
            paths = set()
            for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
                paths.update(different_paths(left_item, right_item, f"{prefix}.{index}"))
            return paths
        return {prefix} if left != right else set()

    assert {path.split(".")[0] for path in different_paths(rl, bc)} == {
        # Full-strength predictive supervision while a dense supervised signal is
        # available to learn the trunk from.
        "predictive_state_coef",
        "predictive_action_coef",
        # BC's own budget: it stops when imitation saturates.
        "total_timesteps",
        # Five entries -- learning_rate, policy_gradient_coef,
        # behavior_cloning_coef, league_fraction, target_kl -- each commented at
        # the point of override in profiles/bc.py.
        "schedule",
    }


def test_token_and_discount_derivations_are_named_and_exact() -> None:
    assert derive_aligned_num_envs(
        rollout_tokens=4_000_000,
        entity_tokens=8,
        num_steps=128,
        num_minibatches=32,
    ) == 3904
    assert derive_aligned_num_envs(
        rollout_tokens=4_000_000,
        entity_tokens=12,
        num_steps=128,
        num_minibatches=32,
    ) == 2592
    assert derive_time_normalized_value(0.99, action_repeat=2) == 0.9801
    assert derive_time_normalized_value(0.95, action_repeat=2) == 0.9025


@pytest.mark.parametrize("name", ("rl", "bc"))
def test_every_profile_checkpoints_on_every_update(name: str) -> None:
    """Save cadence is not a tuning knob any profile owns.

    A save costs a few tens of milliseconds against updates measured in minutes,
    and the writer skips itself rather than queueing when a previous save is
    still running, so a wider interval buys no throughput -- it only sets how
    much progress an interrupted run discards.  Held for BC too: cadence is not
    one of the objective-driven differences ``test_bc_profile`` licenses.
    """

    schedule = resolve_profile(PROFILES[name]).train_config.schedule
    for step in (0, 1, 5_000_000, 500_000_000):
        assert schedule.checkpoint_interval(step) == 1


def test_resolving_one_profile_twice_gives_the_same_configuration() -> None:
    base = resolve_profile(PROFILES["rl"])
    second = resolve_profile(PROFILES["rl"])
    overridden = resolve_profile(
        PROFILES["rl"],
        LaunchOverrides(num_envs=864, microbatch_tokens=20_000),
    )

    assert canonical_data(second.train_config) == canonical_data(base.train_config)
    assert canonical_data(overridden.train_config) != canonical_data(base.train_config)


def test_canonical_serialization_has_a_stable_golden_vector() -> None:
    value = {"z": ["alpha", "beta"], "a": (1, None)}
    assert canonical_json(value) == '{"a":[1,null],"z":["alpha","beta"]}'
    assert fingerprint(value) == "96a880bbbb76a458031417c64ad991bb13f4a18582af7711998ff0253f4c9bd5"
    assert canonical_json(frozenset({"beta", "alpha"})) == '["alpha","beta"]'
    with pytest.raises(ValueError, match="finite"):
        canonical_json(float("nan"))
    with pytest.raises(TypeError, match="unsupported"):
        canonical_json(object())


def test_a_changed_schedule_intent_reaches_the_compiled_schedule() -> None:
    profile = PROFILES["rl"]
    changed_schedule = replace(profile.schedule_spec, entropy_coef=hold(0.006))
    changed = replace(profile, schedule_spec=changed_schedule)

    assert resolve_profile(changed).train_config.schedule.entropy_coef(0) == 0.006
    assert resolve_profile(profile).train_config.schedule.entropy_coef(0) != 0.006


def test_a_keypoint_table_interpolates_holds_and_clamps() -> None:
    linear = compile_keypoints(((0, 1.0, "linear"), (10, 3.0, "hold")))
    assert (linear(-1), linear(0), linear(5), linear(10), linear(11)) == (1.0, 1.0, 2.0, 3.0, 3.0)

    # A row that holds ignores the next value entirely until its step arrives.
    held = compile_keypoints(((0, 1.0, "hold"), (10, 3.0, "hold")))
    assert (held(0), held(9), held(10)) == (1.0, 1.0, 3.0)

    # Exponential interpolation returns the written value at the keypoint, not
    # exp(log(v)): the round trip loses the last bits, and the schedules were
    # tuned under the exact number.
    decay = compile_keypoints(((0, 4.5e-4, "exponential"), (400, 1.5e-4, "hold")))
    assert decay(0) == 4.5e-4
    assert decay(400) == 1.5e-4
    assert decay(200) == 4.5e-4 * (1.5e-4 / 4.5e-4) ** 0.5


def test_a_malformed_keypoint_table_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="at least one keypoint"):
        compile_keypoints((), name="entropy_coef")
    with pytest.raises(ValueError, match="strictly increase"):
        compile_keypoints(((10, 1.0, "hold"), (0, 3.0, "hold")))
    with pytest.raises(ValueError, match="unknown interpolation"):
        compile_keypoints(((0, 1.0, "cosine"),))
    with pytest.raises(ValueError, match="positive value"):
        compile_keypoints(((0, 0.0, "exponential"), (10, 1.0, "hold")))
    with pytest.raises(ValueError, match="needs a number"):
        compile_keypoints(((0, None, "linear"), (10, 1.0, "hold")))
    # A non-numeric value is fine as long as nothing interpolates through it:
    # target_kl is None for the whole of BC.
    assert compile_keypoints(hold(None))(5) is None


def test_resolution_tracks_sources_and_cli_overrides() -> None:
    resolved = resolve_profile(
        PROFILES["rl"],
        LaunchOverrides(num_envs=864, microbatch_tokens=20_000),
    )

    assert resolved.value_sources["train_config.scales.0.num_envs"] == "cli"
    assert resolved.value_sources["train_config.microbatch_tokens"] == "cli"
    assert resolved.value_sources["train_config.gamma"] == "derived"
    assert resolved.value_sources["train_config.component_gammas.ally_win"] == "derived"
    assert resolved.value_sources["model_config.d_model"] == "profile"
    assert resolved.train_config.scales[0].num_envs == 864
    assert resolved.train_config.microbatch_tokens == 20_000

    document = json.loads(
        format_resolved_config(resolve_profile(PROFILES["rl"], LaunchOverrides(864, 20_000)))
    )

    def leaves(value, prefix=""):
        if isinstance(value, dict):
            if not value:
                return {prefix}
            return set().union(
                *(leaves(item, f"{prefix}.{key}" if prefix else key) for key, item in value.items())
            )
        if isinstance(value, list):
            if not value:
                return {prefix}
            return set().union(
                *(
                    leaves(item, f"{prefix}.{index}" if prefix else str(index))
                    for index, item in enumerate(value)
                )
            )
        return {prefix}

    assert set(document["sources"]) == leaves(document["config"])
    assert set(document["sources"].values()) <= {
        "profile",
        "derived",
        "vram-cache",
        "vram-preset",
        "cli",
    }


def test_num_envs_override_recomputes_shards_at_fixed_logical_batch() -> None:
    baseline = resolve_profile(PROFILES["rl"])
    narrower = resolve_profile(PROFILES["rl"], LaunchOverrides(num_envs=864))

    def effective_batch_tokens(resolved) -> int:
        scale = resolved.train_config.scales[0]
        return (
            scale.num_envs
            * resolved.train_config.num_steps
            * (scale.env_config.num_ships + scale.env_config.num_fields)
            * resolved.train_config.rollouts_per_update
        )

    assert baseline.train_config.rollouts_per_update == 3
    assert narrower.train_config.rollouts_per_update == 9
    assert effective_batch_tokens(narrower) == effective_batch_tokens(baseline)
    assert narrower.value_sources["train_config.scales.0.num_envs"] == "cli"
    assert narrower.value_sources["train_config.rollouts_per_update"] == "derived"


@pytest.mark.parametrize("num_envs", (3872, 3904, 23_040))
def test_num_envs_override_rejects_width_that_changes_logical_batch(num_envs: int) -> None:
    with pytest.raises(ValueError, match="fixed logical batch"):
        resolve_profile(PROFILES["rl"], LaunchOverrides(num_envs=num_envs))


def test_equal_explicit_values_keep_value_fingerprint_but_record_cli_source() -> None:
    baseline = resolve_profile(PROFILES["rl"])
    explicit = resolve_profile(
        PROFILES["rl"],
        LaunchOverrides(num_envs=2592, microbatch_tokens=25_000),
    )
    assert canonical_data(explicit.train_config) == canonical_data(baseline.train_config)
    assert explicit.value_sources["train_config.scales.0.num_envs"] == "cli"
    assert explicit.value_sources["train_config.microbatch_tokens"] == "cli"


def test_invalid_launch_override_fails_after_precedence_is_applied() -> None:
    with pytest.raises(ValueError, match="divisible"):
        resolve_profile(PROFILES["rl"], LaunchOverrides(num_envs=1))
    with pytest.raises(ValueError, match="microbatch_tokens"):
        resolve_profile(PROFILES["rl"], LaunchOverrides(microbatch_tokens=0))
    invalid_fixed_width = replace(
        PROFILES["bc"],
        launch=LaunchSizingSpec(num_envs=0),
    )
    with pytest.raises(ValueError, match="num_envs must be positive"):
        resolve_profile(invalid_fixed_width)
    invalid_optimizer = replace(
        PROFILES["rl"],
        clip_coef=-0.1,
    )
    with pytest.raises(ValueError, match="clip_coef"):
        resolve_profile(invalid_optimizer)


def test_fixed_environment_legacy_preset_has_honest_machine_source() -> None:
    """A profile that states a launch width outright says the width is machine-chosen.

    No registered profile still does this — S11 moved BC onto the derived token
    target — so the shape is exercised directly.
    """
    fixed_width = replace(
        PROFILES["rl"],
        logical_batch_tokens=11_943_936,
        launch=LaunchSizingSpec(num_envs=864),
    )
    resolved = resolve_profile(fixed_width)

    assert resolved.value_sources["train_config.scales.0.num_envs"] == "vram-preset"
    assert resolved.value_sources["train_config.rollouts_per_update"] == "derived"
    assert resolved.train_config.rollouts_per_update == 9


def test_format_resolved_config_is_complete_stable_json(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    rendered = format_resolved_config(resolve_profile(PROFILES["rl"]))
    document = json.loads(rendered)

    assert rendered == format_resolved_config(resolve_profile(PROFILES["rl"]))
    assert document["schema_version"] == 1
    assert document["profile"] == "rl"
    assert document["config"]["train_config"]["scales"][0]["num_envs"] == 2592
    assert document["sources"]["train_config.scales.0.num_envs"] == "derived"
    assert list(tmp_path.iterdir()) == []
    assert capsys.readouterr() == ("", "")


def test_resolved_component_discounts_are_deeply_immutable() -> None:
    resolved = resolve_profile(PROFILES["rl"])
    stored_document = resolved_profile_document(resolved)

    with pytest.raises(TypeError):
        resolved.train_config.component_gammas["ally_win"] = 0.5  # type: ignore[index]
    with pytest.raises(TypeError):
        resolved.train_config.component_lambdas["ally_win"] = 0.5  # type: ignore[index]

    assert resolved_profile_document(resolved) == stored_document
    assert json.loads(format_resolved_config(resolve_profile(PROFILES["rl"]))) == stored_document
    serialized = asdict(resolved.train_config)
    assert serialized["component_gammas"] == dict(resolved.train_config.component_gammas)
    assert serialized["component_lambdas"] == dict(resolved.train_config.component_lambdas)


def test_an_overlay_shares_the_bases_values_without_sharing_its_identity() -> None:
    """What BC does not override, it *is* -- and overriding cannot reach back.

    ``replace`` on a frozen dataclass copies, so an overlay holding the base's
    own sub-spec objects is the guarantee that a shared value cannot differ. The
    second half is that the copy is still a copy: nothing done to BC edits RL.
    """

    rl = PROFILES["rl"]
    bc = PROFILES["bc"]

    assert rl is not bc
    assert bc.rewards is rl.rewards
    assert bc.elo_eval is rl.elo_eval
    assert bc.launch is rl.launch
    assert bc.component_gammas_per_tick == rl.component_gammas_per_tick
    assert bc.schedule_spec is not rl.schedule_spec
    assert set(PROFILES) == {"bc", "rl"}
    assert {profile.name for profile in PROFILES.values()} == set(PROFILES)
    with pytest.raises(TypeError):
        rl.component_gammas_per_tick["ally_win"] = 0.5  # type: ignore[index]


def test_only_untransformed_intent_shares_a_name_with_the_resolved_config() -> None:
    """The pass-through set is derived from the two schemas, so naming decides it.

    ``resolve_profile`` copies every field whose name appears on both
    ``ProfileSpec`` and ``TrainConfig``, which is what makes adding a plain
    hyperparameter a two-line change with nothing to edit in the resolver. The
    hazard that buys is a *transformed* value being given the same name on both
    sides -- it would then be copied straight through, and the trainer would read
    the stated intent as if it were the derived result. A per-tick discount would
    silently become a per-decision one.

    So the derived names are pinned here. Adding to this list is a real decision;
    arriving in it by accident is the bug.
    """

    profile_names = {field.name for field in fields(ProfileSpec)}
    resolved_names = {field.name for field in fields(TrainConfig)}

    assert resolved_names - profile_names == {
        "gamma",  # gamma_per_tick raised to action_repeat
        "gae_lambda",  # likewise
        "component_gammas",  # likewise, per component
        "component_lambdas",  # likewise
        "schedule",  # schedule_spec compiled to closures
        "scales",  # env intent plus the derived shard width
        "rollouts_per_update",  # derived from the fixed logical batch
        "microbatch_tokens",  # machine sizing, resolved from launch/VRAM/CLI
    }
    assert set(_PASS_THROUGH) == profile_names & resolved_names
