"""Contracts for VRAM policies, preset rows, and the probe cache.

Nothing here touches an accelerator: this layer decides *what* a VRAM choice
means and which stored measurement applies, and the measuring itself is
``tests/test_vram_probe.py``.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from boost_and_broadside.config.fingerprint import canonical_data
from boost_and_broadside.config.resolve import launch_geometry, resolve_profile
from boost_and_broadside.config.vram import (
    KNOB_TIERS,
    PROBE_VERSION,
    TIER_GUARANTEES,
    VRAM_CACHE_SCHEMA_VERSION,
    VRAM_PRESETS,
    VramCacheEntry,
    VramError,
    VramKnobs,
    VramPolicy,
    VramPreset,
    VramResolution,
    apply_cli_overrides,
    cache_identity,
    cache_path,
    identity_fingerprint,
    launch_overrides,
    parse_vram_policy,
    preset_knobs,
    read_cache,
    resolution_from_cache,
    resolution_from_preset,
    unresolved,
    write_cache_entry,
)
from boost_and_broadside.launch import profile_knobs
from boost_and_broadside.profiles import PROFILES

_DEVICE = {
    "name": "NVIDIA GeForce RTX 4070 Laptop GPU",
    "uuid": "7161aed0-f7cd-9222-9196-257e42144f0d",
    "mig": False,
    "total_memory_bytes": 8_187_281_408,
    "capability": "8.9",
    "multi_processor_count": 36,
}
_SOFTWARE = {"python": "3.13.11", "torch": "2.9.0", "cuda": "12.8", "cudnn": 91002}


def _identity(**changes):
    base = {
        "profile_name": "rl",
        "profile_fingerprint": resolve_profile(PROFILES["rl"]).profile_fingerprint,
        "geometry": launch_geometry(PROFILES["rl"]),
        "compile_mode": "reduce-overhead",
        "device": _DEVICE,
        "software": _SOFTWARE,
    }
    return cache_identity(**{**base, **changes})


def launch_overrides_for(knobs: VramKnobs, source: str = "vram-preset"):
    """The overrides a VRAM decision carrying exactly these knobs would produce."""

    resolution = VramResolution(
        policy="test",
        source=source,  # type: ignore[arg-type]
        status="provisional",
        knobs=knobs,
        applied=knobs,
        identity_fingerprint=None,
    )
    return launch_overrides(resolution, num_envs=None, microbatch_tokens=None)


def _entry(knobs: VramKnobs | None = None, **changes) -> VramCacheEntry:
    identity = _identity(**changes)
    return VramCacheEntry(
        fingerprint=identity_fingerprint(identity),
        created="20260810T120000Z",
        identity=identity,
        knobs=knobs or VramKnobs(1952, 20_000, True),
        measurement={"outcome": "fit", "peak_reserved_bytes": 6_000_000_000},
    )


# ----------------------------------------------------------------------
# Policies
# ----------------------------------------------------------------------


@pytest.mark.parametrize("value", ("auto", "probe", "reprobe", "off", "8", "16", "24", "32"))
def test_every_documented_policy_parses(value: str) -> None:
    policy = parse_vram_policy(value)
    assert str(policy) == value


@pytest.mark.parametrize("value", ("", " auto", "AUTO", "12", "0", "-8", "8gb", "latest", "none"))
def test_undocumented_policies_are_rejected(value: str) -> None:
    with pytest.raises(VramError, match="invalid --vram value"):
        parse_vram_policy(value)


def test_only_auto_and_probe_read_the_cache() -> None:
    assert parse_vram_policy("auto").reads_cache
    assert parse_vram_policy("probe").reads_cache
    # reprobe exists precisely to ignore what is already stored.
    assert not parse_vram_policy("reprobe").reads_cache
    assert not parse_vram_policy("off").reads_cache
    assert parse_vram_policy("reprobe").probes and parse_vram_policy("probe").probes
    assert not parse_vram_policy("auto").probes


# ----------------------------------------------------------------------
# Tiers: what a VRAM decision is allowed to move
# ----------------------------------------------------------------------


def test_vram_may_only_move_tier_one_and_two_knobs() -> None:
    """Tier 3 changes the experiment, so no VRAM knob may belong to it."""

    assert set(KNOB_TIERS) == set(VramKnobs().document())
    assert set(KNOB_TIERS.values()) <= {1, 2}
    assert set(TIER_GUARANTEES) == {1, 2, 3}
    unknown = VramKnobs()
    assert VramKnobs().tiers(unknown) == ()
    assert VramKnobs(grad_checkpoint=True).tiers(unknown) == (1,)
    assert VramKnobs(num_envs=1952).tiers(unknown) == (2,)
    assert VramKnobs(1952, 20_000, True).tiers(unknown) == (1, 2)


def test_a_proposal_that_restates_the_profile_claims_no_tier() -> None:
    """The tier list is the honesty claim, so it may not warn about a change
    nobody made: tier 2's warning is about a different env-stream count and
    minibatch composition, and a knob set to the value it already had produces
    neither."""

    baseline = VramKnobs(3904, 25_000, False)

    assert VramKnobs(3904, 25_000, False).tiers(baseline) == ()
    assert VramKnobs(3904, 25_000, True).tiers(baseline) == (1,)
    assert VramKnobs(1952, 25_000, False).tiers(baseline) == (2,)
    # A knob the proposal does not name never counts, whatever the baseline is.
    assert VramKnobs(num_envs=3904).tiers(baseline) == ()


@pytest.mark.parametrize("name", sorted(PROFILES))
@pytest.mark.parametrize("gigabytes", sorted(VRAM_PRESETS))
def test_no_preset_makes_a_tier_three_change(name: str, gigabytes: int) -> None:
    profile = PROFILES[name]
    geometry = launch_geometry(profile)
    baseline = resolve_profile(profile)
    knobs = preset_knobs(VRAM_PRESETS[gigabytes], geometry)
    sized = resolve_profile(profile, launch_overrides_for(knobs))

    def batch_tokens(resolved) -> int:
        scale = resolved.train_config.scales[0]
        entity = scale.env_config.num_ships + scale.env_config.num_fields
        return (
            scale.num_envs
            * resolved.train_config.num_steps
            * entity
            * resolved.train_config.rollouts_per_update
        )

    assert batch_tokens(sized) == batch_tokens(baseline)
    assert sized.train_config.num_minibatches == baseline.train_config.num_minibatches
    assert sized.train_config.num_steps == baseline.train_config.num_steps
    assert sized.train_config.total_timesteps == baseline.train_config.total_timesteps
    assert canonical_data(sized.env_config) == canonical_data(baseline.env_config)


# ----------------------------------------------------------------------
# Presets
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_the_eight_gigabyte_row_is_exactly_the_shipped_launch(name: str) -> None:
    """The measured row must change nothing, or `--vram 8` is not what ships."""

    profile = PROFILES[name]
    geometry = launch_geometry(profile)
    knobs = preset_knobs(VRAM_PRESETS[8], geometry)

    assert knobs.num_envs == geometry.default_num_envs
    assert knobs.microbatch_tokens == geometry.default_microbatch_tokens
    assert knobs.grad_checkpoint == profile.model_config.grad_checkpoint

    baseline = resolve_profile(profile)
    sized = resolve_profile(profile, launch_overrides_for(knobs))
    assert canonical_data(sized.train_config) == canonical_data(baseline.train_config)
    assert canonical_data(sized.model_config) == canonical_data(baseline.model_config)
    assert sized.resolved_config_fingerprint == baseline.resolved_config_fingerprint
    # And it must not claim it moved anything either.
    assert knobs.tiers(profile_knobs(baseline)) == ()


def test_a_bigger_row_holds_more_of_the_fixed_batch_resident() -> None:
    geometry = launch_geometry(PROFILES["rl"])
    widths = {
        gigabytes: preset_knobs(VRAM_PRESETS[gigabytes], geometry).num_envs
        for gigabytes in sorted(VRAM_PRESETS)
    }
    assert widths == {8: 3904, 16: 5856, 24: 11712, 32: 11712}
    shards = [
        geometry.aligned_logical_batch_tokens // geometry.rollout_tokens(width)
        for width in widths.values()
    ]
    assert shards == [3, 2, 1, 1]


def test_a_profile_without_an_intermediate_width_keeps_the_narrower_one() -> None:
    """rl-fields has no 2-shard split, so its 16 GB row is honestly its 8 GB row."""

    geometry = launch_geometry(PROFILES["rl-fields"])
    assert [width for width, _ in geometry.shard_widths()] == [7776, 2592, 864, 288, 96, 32]
    assert preset_knobs(VRAM_PRESETS[16], geometry).num_envs == 2592
    assert preset_knobs(VRAM_PRESETS[24], geometry).num_envs == 7776


def test_every_shard_width_preserves_the_fixed_logical_batch() -> None:
    for profile in PROFILES.values():
        geometry = launch_geometry(profile)
        for num_envs, shards in geometry.shard_widths():
            assert num_envs % geometry.num_minibatches == 0
            assert (
                geometry.rollout_tokens(num_envs) * shards
                == geometry.aligned_logical_batch_tokens
            )
        assert (geometry.default_num_envs, geometry.default_rollouts_per_update) in (
            geometry.shard_widths()
        )


def test_a_row_no_width_can_satisfy_fails_loudly() -> None:
    impossible = replace(VRAM_PRESETS[8], max_rollout_tokens=1)
    with pytest.raises(VramError, match="no shard width preserves"):
        preset_knobs(impossible, launch_geometry(PROFILES["rl"]))


def test_only_a_measured_row_names_the_device_it_was_measured_on() -> None:
    for preset in VRAM_PRESETS.values():
        if preset.measured_on is None:
            assert "provisional" in preset.basis
        else:
            assert preset.basis and "provisional" not in preset.basis
    assert [gigabytes for gigabytes, row in VRAM_PRESETS.items() if row.measured_on] == [8]


def test_applying_a_preset_is_provisional_even_when_its_row_was_measured() -> None:
    """A measurement belongs to the card it was taken on, not to this one."""

    resolution = resolution_from_preset(
        VramPolicy("preset", 8), VRAM_PRESETS[8], launch_geometry(PROFILES["rl"])
    )
    assert resolution.status == "provisional"
    assert resolution.source == "vram-preset"
    assert any("run --vram probe" in note for note in resolution.notes)
    assert VRAM_PRESETS[8].measured_on in resolution.notes[0]


def test_only_a_cache_entry_for_this_machine_is_called_measured() -> None:
    measured = resolution_from_cache(VramPolicy("auto"), _entry())
    assert measured.status == "measured"
    assert measured.source == "vram-cache"
    assert unresolved(VramPolicy("off")).status == "unresolved"
    assert unresolved(VramPolicy("off")).source is None


# ----------------------------------------------------------------------
# Cache identity and invalidation
# ----------------------------------------------------------------------


def test_identity_is_stable_and_order_independent() -> None:
    assert identity_fingerprint(_identity()) == identity_fingerprint(_identity())
    assert len(identity_fingerprint(_identity())) == 64
    assert _identity()["probe_version"] == PROBE_VERSION
    assert _identity()["autocast_dtype"] == "bfloat16"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("compile_mode", None),
        ("compile_mode", "max-autotune"),
        ("profile_name", "bc"),
        ("profile_fingerprint", "0" * 64),
    ),
)
def test_a_changed_launch_question_invalidates_the_entry(field: str, value) -> None:
    assert identity_fingerprint(_identity(**{field: value})) != identity_fingerprint(_identity())


@pytest.mark.parametrize(
    "change",
    (
        {"uuid": "0000aed0-f7cd-9222-9196-257e42144f0d"},
        {"total_memory_bytes": 16_000_000_000},
        {"mig": True},
        {"name": "NVIDIA A100-SXM4-40GB"},
        {"capability": "9.0"},
        {"multi_processor_count": 108},
    ),
)
def test_a_changed_device_invalidates_the_entry(change: dict) -> None:
    other = identity_fingerprint(_identity(device={**_DEVICE, **change}))
    assert other != identity_fingerprint(_identity())


@pytest.mark.parametrize("change", ({"torch": "2.10.0"}, {"cuda": "13.0"}, {"cudnn": 91100}))
def test_a_changed_software_stack_invalidates_the_entry(change: dict) -> None:
    other = identity_fingerprint(_identity(software={**_SOFTWARE, **change}))
    assert other != identity_fingerprint(_identity())


def test_the_geometry_a_measurement_answered_for_is_part_of_its_identity() -> None:
    fields = launch_geometry(PROFILES["rl"])
    narrowed = replace(
        fields, aligned_logical_batch_tokens=fields.aligned_logical_batch_tokens // 2
    )
    assert identity_fingerprint(_identity(geometry=narrowed)) != identity_fingerprint(_identity())


# ----------------------------------------------------------------------
# Cache file
# ----------------------------------------------------------------------


def test_cache_round_trips_and_keeps_unrelated_entries(tmp_path: Path) -> None:
    path = tmp_path / ".vram.json"
    assert read_cache(path) == {}

    first = _entry()
    second = _entry(profile_name="bc")
    write_cache_entry(path, first)
    write_cache_entry(path, second)

    stored = read_cache(path)
    assert set(stored) == {first.fingerprint, second.fingerprint}
    assert stored[first.fingerprint].knobs == first.knobs
    assert stored[first.fingerprint].identity == first.identity
    # Nothing is left behind, and the file is deterministic.
    assert sorted(item.name for item in tmp_path.iterdir()) == [".vram.json"]
    document = json.loads(path.read_text())
    assert document["schema_version"] == VRAM_CACHE_SCHEMA_VERSION
    assert [entry["fingerprint"] for entry in document["entries"]] == sorted(stored)


def test_reprobing_replaces_the_entry_for_the_same_machine(tmp_path: Path) -> None:
    path = tmp_path / ".vram.json"
    write_cache_entry(path, _entry(VramKnobs(3904, 25_000, False)))
    write_cache_entry(path, _entry(VramKnobs(1952, 12_500, True)))

    stored = read_cache(path)
    assert len(stored) == 1
    assert next(iter(stored.values())).knobs == VramKnobs(1952, 12_500, True)


def test_a_failed_write_leaves_the_previous_cache_intact(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / ".vram.json"
    write_cache_entry(path, _entry())
    before = path.read_text()

    monkeypatch.setattr(
        "boost_and_broadside.config.vram.os.replace",
        lambda *_: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(OSError, match="disk full"):
        write_cache_entry(path, _entry(profile_name="bc"))

    assert path.read_text() == before
    assert sorted(item.name for item in tmp_path.iterdir()) == [".vram.json"]


@pytest.mark.parametrize(
    ("content", "match"),
    (
        ("not json at all", "could not be read"),
        ('{"schema_version": 99, "entries": []}', "schema version 99"),
        ('{"schema_version": 1}', "no entry list"),
        ('{"schema_version": 1, "entries": [{"fingerprint": "a"}]}', "malformed"),
        ('["a list"]', "schema version None"),
    ),
)
def test_a_cache_that_cannot_be_understood_is_an_error(
    tmp_path: Path, content: str, match: str
) -> None:
    """Silently resizing the launch because a recomputable file broke is the
    substitution this whole system exists to prevent."""

    path = tmp_path / ".vram.json"
    path.write_text(content)
    with pytest.raises(VramError, match=match):
        read_cache(path)


@pytest.mark.parametrize(
    "content", ("not json at all", '{"schema_version": 99, "entries": []}', '{"schema_version": 1}')
)
def test_a_fresh_measurement_replaces_a_cache_it_cannot_read(
    tmp_path: Path, content: str
) -> None:
    """The recovery ``read_cache`` names has to work.

    It tells the user to launch with ``--vram reprobe``; a reprobe reaches the
    write only after measuring the card, so raising there would make the advice
    circular and discard twenty minutes of measurement per candidate.
    """

    path = tmp_path / ".vram.json"
    path.write_text(content)
    entry = _entry(VramKnobs(1952, 12_500, True))
    lines: list[str] = []

    write_cache_entry(path, entry, report=lines.append)

    stored = read_cache(path)
    assert list(stored) == [entry.fingerprint]
    assert stored[entry.fingerprint].knobs == VramKnobs(1952, 12_500, True)
    assert len(lines) == 1 and "replacing the unreadable cache" in lines[0]
    assert sorted(item.name for item in tmp_path.iterdir()) == [".vram.json"]


def test_an_entry_with_an_unknown_knob_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / ".vram.json"
    entry = _entry().document()
    entry["knobs"]["num_minibatches"] = 4
    path.write_text(json.dumps({"schema_version": 1, "entries": [entry]}))
    with pytest.raises(VramError, match="unknown VRAM knobs"):
        read_cache(path)


def test_the_cache_lives_beside_the_working_tree(tmp_path: Path) -> None:
    assert cache_path(tmp_path) == tmp_path / ".vram.json"
    assert cache_path().name == ".vram.json"


# ----------------------------------------------------------------------
# Precedence
# ----------------------------------------------------------------------


def test_explicit_overrides_outrank_a_vram_proposal_and_say_so() -> None:
    proposal = resolution_from_cache(VramPolicy("auto"), _entry(VramKnobs(5856, 37_500, True)))
    applied = apply_cli_overrides(proposal, num_envs=1952, microbatch_tokens=None)

    assert applied.knobs.num_envs == 5856, "the proposal is still recorded"
    assert applied.applied.num_envs is None, "but it is not what ran"
    assert applied.applied.microbatch_tokens == 37_500
    assert any("overrides the vram-cache width 5856" in note for note in applied.notes)

    overrides = launch_overrides(applied, num_envs=1952, microbatch_tokens=None)
    assert (overrides.num_envs, overrides.num_envs_source) == (1952, "cli")
    assert (overrides.microbatch_tokens, overrides.microbatch_tokens_source) == (
        37_500,
        "vram-cache",
    )
    assert (overrides.grad_checkpoint, overrides.grad_checkpoint_source) == (True, "vram-cache")


def test_an_override_equal_to_the_proposal_records_no_conflict() -> None:
    proposal = resolution_from_cache(VramPolicy("auto"), _entry(VramKnobs(5856, 37_500, False)))
    applied = apply_cli_overrides(proposal, num_envs=5856, microbatch_tokens=37_500)
    assert applied.notes == proposal.notes
    overrides = launch_overrides(applied, num_envs=5856, microbatch_tokens=37_500)
    assert overrides.num_envs_source == "cli"


def test_an_unresolved_policy_changes_nothing() -> None:
    resolution = unresolved(VramPolicy("off"), "--vram off")
    overrides = launch_overrides(resolution, num_envs=None, microbatch_tokens=None)
    assert overrides.num_envs is None
    assert overrides.microbatch_tokens is None
    assert overrides.grad_checkpoint is None

    baseline = resolve_profile(PROFILES["rl"])
    unchanged = resolve_profile(PROFILES["rl"], overrides)
    assert unchanged.resolved_config_fingerprint == baseline.resolved_config_fingerprint
    assert unchanged.value_sources["train_config.scales.0.num_envs"] == "derived"
    assert unchanged.value_sources["model_config.grad_checkpoint"] == "profile"


def test_a_vram_proposal_is_recorded_as_its_own_source() -> None:
    proposal = resolution_from_cache(VramPolicy("auto"), _entry(VramKnobs(1952, 20_000, True)))
    resolved = resolve_profile(
        PROFILES["rl"], launch_overrides(proposal, num_envs=None, microbatch_tokens=None)
    )
    assert resolved.value_sources["train_config.scales.0.num_envs"] == "vram-cache"
    assert resolved.value_sources["train_config.microbatch_tokens"] == "vram-cache"
    assert resolved.value_sources["model_config.grad_checkpoint"] == "vram-cache"
    assert resolved.model_config.grad_checkpoint is True
    assert resolved.train_config.rollouts_per_update == 6


def test_the_resolution_document_states_the_guarantee_of_what_it_moved() -> None:
    baseline = VramKnobs(3904, 25_000, False)
    moved = VramKnobs(1952, 20_000, True)
    document = resolution_from_cache(VramPolicy("auto"), _entry(moved)).document(baseline, moved)
    assert document["status"] == "measured"
    assert document["source"] == "vram-cache"
    assert set(document["tiers"]) == {"1", "2"}
    assert document["tiers"]["2"] == TIER_GUARANTEES[2]
    assert "3" not in document["tiers"]

    restated = resolution_from_cache(VramPolicy("auto"), _entry(baseline)).document(
        baseline, baseline
    )
    assert restated["applied"] == baseline.document()
    assert restated["tiers"] == {}

    off = unresolved(VramPolicy("off"), "note").document(baseline, baseline)
    assert off["tiers"] == {}
    assert off["notes"] == ["note"]


def test_a_preset_row_is_a_value_not_a_mutable_table() -> None:
    with pytest.raises(Exception):
        VRAM_PRESETS[8].gigabytes = 9  # type: ignore[misc]
    assert isinstance(VRAM_PRESETS[8], VramPreset)
