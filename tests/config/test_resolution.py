"""Contracts for the profile-intent and resolved-configuration boundary."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from boost_and_broadside.config.fingerprint import canonical_data, canonical_json, fingerprint
from boost_and_broadside.config.resolve import (
    LaunchOverrides,
    derive_aligned_num_envs,
    derive_time_normalized_value,
    resolve_profile,
)
from boost_and_broadside.config.schedule_spec import compile_schedule, constant_spec, linear_spec
from boost_and_broadside.config.schema import LaunchSizingSpec
from boost_and_broadside.config.service import format_resolved_profile, resolved_profile_document
from boost_and_broadside.profiles import PROFILES

_ROOT = Path(__file__).resolve().parents[2]
_SNAPSHOTS = _ROOT / "tests" / "fixtures" / "mode_refactor"
_PROFILE_MODULES = (
    _ROOT / "src" / "boost_and_broadside" / "profiles" / "rl.py",
    _ROOT / "src" / "boost_and_broadside" / "profiles" / "rl_fields.py",
    _ROOT / "src" / "boost_and_broadside" / "profiles" / "bc.py",
)


@pytest.mark.parametrize("name", ("rl", "rl-fields"))
def test_resolved_profiles_match_s01_snapshots(name: str) -> None:
    """BC is deliberately absent: S11 corrected it away from its S01 evidence.

    ``tests/config/test_bc_profile.py`` pins the corrected profile and the exact
    set of values that correction changed.
    """
    expected = json.loads((_SNAPSHOTS / f"{name}.json").read_text())
    resolved = resolve_profile(PROFILES[name])

    assert canonical_data(resolved.ship_config) == expected["ship_config"]
    assert canonical_data(resolved.model_config) == expected["model_config"]
    assert canonical_data(resolved.train_config) == expected["train_config"]


def test_rl_fields_resolved_diff_contains_only_named_field_intent() -> None:
    rl = canonical_data(resolve_profile(PROFILES["rl"]).train_config)
    fields = canonical_data(resolve_profile(PROFILES["rl-fields"]).train_config)

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

    # The live gauge is defined rather than fitted per environment, so S12
    # removed the two rating differences that used to be here: what separates
    # the two profiles is now field intent and nothing else.
    assert different_paths(rl, fields) == {
        "field_map",
        "rewards.field_damage_taken_weight",
        "rewards.field_death_weight",
        "scales.0.env_config.num_fields",
        "scales.0.num_envs",
    }


def test_registered_profile_modules_do_not_import_each_other() -> None:
    module_names = {
        "boost_and_broadside.profiles.rl",
        "boost_and_broadside.profiles.rl_fields",
        "boost_and_broadside.profiles.bc",
    }
    for path in _PROFILE_MODULES:
        tree = ast.parse(path.read_text(), filename=str(path))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        imported.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert imported.isdisjoint(module_names), f"{path.name} imports another profile: {imported}"
    assert not (_ROOT / "runs").exists()


def test_deleted_runs_profile_path_has_no_live_references() -> None:
    historical_documents = {
        _ROOT / "docs" / "internal" / "mode-refactor-plan.md",
        _ROOT / "docs" / "internal" / "mode-refactor-status.md",
    }
    candidates = [
        _ROOT / "src" / "boost_and_broadside" / "cli.py",
        _ROOT / "src" / "boost_and_broadside" / "cli_commands.py",
        _ROOT / "README.md",
        _ROOT / "STYLE_GUIDE.md",
        _ROOT / "pyproject.toml",
    ]
    for root in (_ROOT / "src", _ROOT / "scripts", _ROOT / "docs"):
        candidates.extend(
            path
            for path in root.rglob("*")
            if path.suffix in {".md", ".py", ".toml"}
            and path not in historical_documents
        )

    stale_reference = re.compile(r"\b(?:from|import)\s+runs\b|\bruns/")
    offenders = {
        str(path.relative_to(_ROOT)): sorted(set(stale_reference.findall(path.read_text())))
        for path in candidates
        if stale_reference.search(path.read_text())
    }
    assert offenders == {}


def test_config_foundation_has_no_runtime_engine_dependencies() -> None:
    roots = (
        _ROOT / "src" / "boost_and_broadside" / "config",
        _ROOT / "src" / "boost_and_broadside" / "profiles",
    )
    forbidden = (
        "boost_and_broadside.env",
        "boost_and_broadside.modes",
        "boost_and_broadside.train",
    )
    for path in (path for root in roots for path in root.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        modules = [
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        ]
        modules.extend(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert not [module for module in modules if module.startswith(forbidden)], path


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


def test_fingerprints_are_canonical_and_separate_intent_from_launch() -> None:
    base = resolve_profile(PROFILES["rl"])
    overridden = resolve_profile(
        PROFILES["rl"],
        LaunchOverrides(num_envs=1952, microbatch_tokens=20_000),
    )

    assert fingerprint({"b": 2, "a": 1}) == fingerprint({"a": 1, "b": 2})
    assert base.profile_fingerprint == overridden.profile_fingerprint
    assert base.resolved_config_fingerprint != overridden.resolved_config_fingerprint
    second = resolve_profile(PROFILES["rl"])
    assert canonical_data(second.train_config) == canonical_data(base.train_config)
    assert second.profile_fingerprint == base.profile_fingerprint
    assert second.resolved_config_fingerprint == base.resolved_config_fingerprint


def test_canonical_serialization_has_a_stable_golden_vector() -> None:
    value = {"z": ["alpha", "beta"], "a": (1, None)}
    assert canonical_json(value) == '{"a":[1,null],"z":["alpha","beta"]}'
    assert fingerprint(value) == "96a880bbbb76a458031417c64ad991bb13f4a18582af7711998ff0253f4c9bd5"
    assert canonical_json(frozenset({"beta", "alpha"})) == '["alpha","beta"]'
    with pytest.raises(ValueError, match="finite"):
        canonical_json(float("nan"))
    with pytest.raises(TypeError, match="unsupported"):
        canonical_json(object())


def test_current_profile_and_resolved_fingerprints_are_stable() -> None:
    # All six moved in S12: replacing each profile's fitted reference ladder and
    # random rating with the derived live gauge is a semantic change to what the
    # run is rated against, so both fingerprints are expected to differ from the
    # values recorded through S11.
    expected = {
        "rl": (
            "9f4baf830c22ffb198b2ffd67412113376761d236f8863bc8bdefc29be933323",
            "882ed9ba23a1bf1cd0b3030966d2e30358e7ee7c26d5186580ba5ab10d322ee3",
        ),
        "rl-fields": (
            "cdd020cf01d8e0a16425ea9a34c00ba84cdb4b549f8345a6955e0106605ba047",
            "1a5a3c43a53401671fa6b446c2e6163c29a154cad3626b7cb617c497e25bea08",
        ),
        "bc": (
            "138544ad41438242bbe795025773970389ba161e522f1a8959cef54c214c5a19",
            "b91a4ef73a92309ddabbc34ce58f485b3497890fafc9a6db4541174e209c2a6d",
        ),
    }
    for name, fingerprints in expected.items():
        resolved = resolve_profile(PROFILES[name])
        assert (resolved.profile_fingerprint, resolved.resolved_config_fingerprint) == fingerprints


def test_profile_fingerprint_includes_declarative_schedule_intent() -> None:
    profile = PROFILES["rl"]
    changed_schedule = replace(
        profile.objective.schedule,
        entropy_coef=constant_spec(0.006),
    )
    changed = replace(
        profile,
        objective=replace(profile.objective, schedule=changed_schedule),
    )
    changed_fingerprint = resolve_profile(changed).profile_fingerprint
    assert changed_fingerprint != resolve_profile(profile).profile_fingerprint


def test_profile_fingerprint_excludes_legacy_machine_launch_preset() -> None:
    profile = PROFILES["rl"]
    changed = replace(
        profile,
        launch_defaults=replace(profile.launch_defaults, rollout_tokens=3_000_000),
    )
    baseline = resolve_profile(profile)
    other_machine = resolve_profile(changed)
    assert other_machine.profile_fingerprint == baseline.profile_fingerprint
    assert other_machine.resolved_config_fingerprint != baseline.resolved_config_fingerprint


def test_profile_fingerprint_excludes_gradient_checkpointing() -> None:
    profile = PROFILES["rl"]
    changed = replace(
        profile,
        model_config=replace(
            profile.model_config,
            grad_checkpoint=not profile.model_config.grad_checkpoint,
        ),
    )
    baseline = resolve_profile(profile)
    other_machine = resolve_profile(changed)

    assert canonical_data(other_machine.train_config) == canonical_data(baseline.train_config)
    assert other_machine.profile_fingerprint == baseline.profile_fingerprint
    assert other_machine.resolved_config_fingerprint != baseline.resolved_config_fingerprint


def test_declarative_schedule_validation_and_boundaries() -> None:
    with pytest.raises(ValueError, match="at least two"):
        linear_spec((0, 1.0))
    with pytest.raises(ValueError, match="strictly increasing"):
        linear_spec((10, 1.0), (0, 3.0))
    schedule = linear_spec((0, 1.0), (10, 3.0))
    runtime = compile_schedule(schedule)
    assert (runtime(-1), runtime(5), runtime(11)) == (1.0, 2.0, 3.0)


def test_resolution_tracks_sources_and_cli_overrides() -> None:
    resolved = resolve_profile(
        PROFILES["rl"],
        LaunchOverrides(num_envs=1952, microbatch_tokens=20_000),
    )

    assert resolved.value_sources["train_config.scales.0.num_envs"] == "cli"
    assert resolved.value_sources["train_config.microbatch_tokens"] == "cli"
    assert resolved.value_sources["train_config.gamma"] == "derived"
    assert resolved.value_sources["train_config.component_gammas.ally_win"] == "derived"
    assert resolved.value_sources["model_config.d_model"] == "profile"
    assert resolved.train_config.scales[0].num_envs == 1952
    assert resolved.train_config.microbatch_tokens == 20_000

    document = json.loads(format_resolved_profile("rl", LaunchOverrides(1952, 20_000)))

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
    half_width = resolve_profile(PROFILES["rl"], LaunchOverrides(num_envs=1952))

    def effective_batch_tokens(resolved) -> int:
        scale = resolved.train_config.scales[0]
        return (
            scale.num_envs
            * resolved.train_config.num_steps
            * (scale.env_config.num_ships + scale.env_config.num_fields)
            * resolved.train_config.rollouts_per_update
        )

    assert baseline.train_config.rollouts_per_update == 3
    assert half_width.train_config.rollouts_per_update == 6
    assert effective_batch_tokens(half_width) == effective_batch_tokens(baseline)
    assert half_width.value_sources["train_config.scales.0.num_envs"] == "cli"
    assert half_width.value_sources["train_config.rollouts_per_update"] == "derived"


@pytest.mark.parametrize("num_envs", (3872, 7776, 23_040))
def test_num_envs_override_rejects_width_that_changes_logical_batch(num_envs: int) -> None:
    with pytest.raises(ValueError, match="fixed logical batch"):
        resolve_profile(PROFILES["rl"], LaunchOverrides(num_envs=num_envs))


def test_equal_explicit_values_keep_value_fingerprint_but_record_cli_source() -> None:
    baseline = resolve_profile(PROFILES["rl"])
    explicit = resolve_profile(
        PROFILES["rl"],
        LaunchOverrides(num_envs=3904, microbatch_tokens=25_000),
    )
    assert explicit.resolved_config_fingerprint == baseline.resolved_config_fingerprint
    assert explicit.value_sources["train_config.scales.0.num_envs"] == "cli"
    assert explicit.value_sources["train_config.microbatch_tokens"] == "cli"


def test_invalid_launch_override_fails_after_precedence_is_applied() -> None:
    with pytest.raises(ValueError, match="divisible"):
        resolve_profile(PROFILES["rl"], LaunchOverrides(num_envs=1))
    with pytest.raises(ValueError, match="microbatch_tokens"):
        resolve_profile(PROFILES["rl"], LaunchOverrides(microbatch_tokens=0))
    invalid_fixed_width = replace(
        PROFILES["bc"],
        launch_defaults=LaunchSizingSpec(num_envs=0),
    )
    with pytest.raises(ValueError, match="num_envs must be positive"):
        resolve_profile(invalid_fixed_width)
    invalid_optimizer = replace(
        PROFILES["rl"],
        optimizer=replace(PROFILES["rl"].optimizer, clip_coef=-0.1),
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
        rollout=replace(PROFILES["rl"].rollout, logical_batch_tokens=11_993_088),
        launch_defaults=LaunchSizingSpec(num_envs=3904),
    )
    resolved = resolve_profile(fixed_width)

    assert resolved.value_sources["train_config.scales.0.num_envs"] == "vram-preset"
    assert resolved.value_sources["train_config.rollouts_per_update"] == "derived"
    assert resolved.train_config.rollouts_per_update == 3


def test_format_resolved_profile_is_complete_stable_json(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    rendered = format_resolved_profile("rl")
    document = json.loads(rendered)

    assert rendered == format_resolved_profile("rl")
    assert document["schema_version"] == 1
    assert document["profile"] == "rl"
    assert document["config"]["train_config"]["scales"][0]["num_envs"] == 3904
    assert document["sources"]["train_config.scales.0.num_envs"] == "derived"
    assert len(document["profile_fingerprint"]) == 64
    assert len(document["resolved_config_fingerprint"]) == 64
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
    assert json.loads(format_resolved_profile("rl")) == stored_document
    serialized = asdict(resolved.train_config)
    assert serialized["component_gammas"] == dict(resolved.train_config.component_gammas)
    assert serialized["component_lambdas"] == dict(resolved.train_config.component_lambdas)


def test_profiles_are_independent_values_even_when_their_intent_matches() -> None:
    rl = PROFILES["rl"]
    fields = PROFILES["rl-fields"]

    assert rl is not fields
    assert rl.objective.schedule is not fields.objective.schedule
    assert replace(fields.environment, num_fields=0) == rl.environment
    assert set(PROFILES) == {"bc", "rl", "rl-fields"}
    assert {profile.name for profile in PROFILES.values()} == set(PROFILES)
    with pytest.raises(TypeError):
        rl.discounts.component_gammas_per_tick["ally_win"] = 0.5  # type: ignore[index]
