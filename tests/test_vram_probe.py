"""Contracts for VRAM probing, cache lookup, and complete launch composition.

The probe's own measurement needs a CUDA device, so these tests drive it with an
injected candidate runner and an injected device identity.  What is verified
here is everything around the measurement: which candidates are tried and in
what order, that each one runs in a fresh subprocess rather than in this
allocator, which stored entry applies to a machine, and that the resolved launch
records where every value came from.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from boost_and_broadside import execution, vram_probe
from boost_and_broadside.config.resolve import launch_geometry, resolve_profile
from boost_and_broadside.config.vram import (
    VRAM_PRESETS,
    VramError,
    VramKnobs,
    VramPolicy,
    preset_knobs,
    read_cache,
    write_cache_entry,
)
from boost_and_broadside.errors import UserFacingError
from boost_and_broadside.launch import resolve_training_launch
from boost_and_broadside.profiles import PROFILES
from boost_and_broadside.vram_probe import (
    OOM_EXIT_CODE,
    ProbeOutcome,
    _child_command,
    candidate_knobs,
    probe_profile,
    resolve_vram,
)

_EIGHT_GIB = 8_187_281_408
_IDENTITY = {
    "name": "NVIDIA GeForce RTX 4070 Laptop GPU",
    "uuid": "7161aed0-f7cd-9222-9196-257e42144f0d",
    "mig": False,
    "total_memory_bytes": _EIGHT_GIB,
    "capability": "8.9",
    "multi_processor_count": 36,
}
_RL_FINGERPRINT = resolve_profile(PROFILES["rl"]).profile_fingerprint


@pytest.fixture
def fake_cuda(monkeypatch):
    """Present a fixed CUDA device without needing one, or touching one."""

    monkeypatch.setattr(execution, "resolve_device", lambda requested: requested)
    monkeypatch.setattr(vram_probe, "device_identity", lambda device: dict(_IDENTITY))
    monkeypatch.setattr(
        vram_probe, "software_identity", lambda: {"python": "3.13.11", "torch": "2.9.0"}
    )


def _runner(*fitting: VramKnobs):
    """A candidate runner that accepts only the named configurations."""

    attempted: list[VramKnobs] = []

    def run(knobs: VramKnobs) -> ProbeOutcome:
        attempted.append(knobs)
        fit = knobs in fitting
        return ProbeOutcome(
            knobs=knobs,
            fit=fit,
            measurement={
                "outcome": "fit" if fit else "oom",
                "peak_reserved_bytes": 6_000_000_000,
            },
        )

    run.attempted = attempted  # type: ignore[attr-defined]
    return run


# ----------------------------------------------------------------------
# The candidate ladder
# ----------------------------------------------------------------------


def test_the_ladder_starts_at_the_largest_row_the_card_could_hold() -> None:
    geometry = launch_geometry(PROFILES["rl"])
    ladder = candidate_knobs(geometry, total_memory_bytes=_EIGHT_GIB)

    # An "8 GB" card reports about 7.6 GiB; the row still matches it, and no
    # larger row is attempted.
    assert ladder[0] == preset_knobs(VRAM_PRESETS[8], geometry)
    assert all(knobs.num_envs <= 3904 for knobs in ladder)
    # Below the smallest row, the exact knob comes before the sampling one.
    assert ladder[1] == VramKnobs(3904, 25_000, True)
    assert [knobs.num_envs for knobs in ladder] == [3904, 3904, 1952, 192]
    assert all(knobs.grad_checkpoint for knobs in ladder[1:])


def test_a_larger_card_tries_larger_rows_first() -> None:
    geometry = launch_geometry(PROFILES["rl"])
    ladder = candidate_knobs(geometry, total_memory_bytes=32 * 1024**3)
    assert [knobs.num_envs for knobs in ladder][:4] == [11712, 11712, 5856, 3904]
    assert ladder[0].microbatch_tokens == VRAM_PRESETS[32].microbatch_tokens


def test_the_ladder_is_bounded_and_every_candidate_is_a_valid_launch() -> None:
    for name, profile in PROFILES.items():
        geometry = launch_geometry(profile)
        ladder = candidate_knobs(geometry, total_memory_bytes=32 * 1024**3)
        assert 0 < len(ladder) <= 8, name
        assert len(set(ladder)) == len(ladder), f"{name} repeats a candidate"
        for knobs in ladder:
            # Each one has to survive the resolver, or the probe would spend an
            # hour measuring configurations that could never launch.
            resolved = resolve_profile(
                profile,
                _overrides(knobs),
            )
            assert resolved.train_config.scales[0].num_envs == knobs.num_envs
            assert resolved.model_config.grad_checkpoint == knobs.grad_checkpoint


def _overrides(knobs: VramKnobs):
    from boost_and_broadside.config.resolve import LaunchOverrides

    return LaunchOverrides(
        num_envs=knobs.num_envs,
        microbatch_tokens=knobs.microbatch_tokens,
        grad_checkpoint=knobs.grad_checkpoint,
    )


# ----------------------------------------------------------------------
# Probing
# ----------------------------------------------------------------------


def test_the_first_candidate_that_fits_wins_and_the_rest_are_recorded() -> None:
    geometry = launch_geometry(PROFILES["rl"])
    winner = VramKnobs(3904, 25_000, True)
    run = _runner(winner)

    entry, attempts = probe_profile(
        "rl",
        device="cuda",
        compile_mode=None,
        profile_fingerprint=_RL_FINGERPRINT,
        runner=run,
        identity=_IDENTITY,
    )

    assert entry.knobs == winner
    assert [outcome.knobs for outcome in attempts] == run.attempted
    assert attempts[0].knobs == preset_knobs(VRAM_PRESETS[8], geometry)
    assert not attempts[0].fit and attempts[-1].fit
    assert entry.measurement["candidates_tried"] == 2
    assert entry.measurement["rejected"] == [
        {"knobs": attempts[0].knobs.document(), "outcome": "oom"}
    ]
    assert entry.identity["device"] == _IDENTITY


def test_a_card_that_fits_nothing_fails_with_every_rejection_named() -> None:
    with pytest.raises(VramError, match="no probed configuration fit"):
        probe_profile(
            "rl",
            device="cuda",
            compile_mode=None,
            profile_fingerprint=_RL_FINGERPRINT,
            runner=_runner(),
            identity=_IDENTITY,
        )


def test_each_candidate_is_measured_in_a_fresh_interpreter() -> None:
    """The parent must never allocate a candidate: an OOM would leave its
    allocator fragmented for the next attempt."""

    command = _child_command(
        VramKnobs(3904, 25_000, True), profile="rl", device="cuda:0", compile_mode=None
    )
    assert command[:3] == [sys.executable, "-m", "boost_and_broadside.vram_probe"]
    assert command[3:] == [
        "--profile",
        "rl",
        "--device",
        "cuda:0",
        "--num-envs",
        "3904",
        "--microbatch-tokens",
        "25000",
        "--compile",
        "none",
        "--grad-checkpoint",
    ]
    assert "--grad-checkpoint" not in _child_command(
        VramKnobs(3904, 25_000, False), profile="rl", device="cuda", compile_mode="default"
    )


def test_the_child_refuses_a_device_it_cannot_measure(tmp_path: Path) -> None:
    """A real subprocess, so the module entry point itself is covered."""

    completed = subprocess.run(
        _child_command(
            VramKnobs(3904, 25_000, False), profile="rl", device="cpu", compile_mode=None
        ),
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    assert completed.returncode == 1, completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["outcome"] == "error"
    assert "needs a CUDA device" in result["error"]
    assert result["knobs"] == {
        "num_envs": 3904,
        "microbatch_tokens": 25_000,
        "grad_checkpoint": False,
    }
    assert list(tmp_path.iterdir()) == []


def test_a_child_that_reports_nothing_is_an_error_not_a_rejection(monkeypatch) -> None:
    class _Completed:
        returncode = 1
        stdout = ""
        stderr = "Traceback...\nImportError: boom"

    monkeypatch.setattr(subprocess, "run", lambda *_, **__: _Completed())
    runner = vram_probe.subprocess_runner(profile="rl", device="cuda", compile_mode=None)
    with pytest.raises(VramError, match="ImportError: boom"):
        runner(VramKnobs(3904, 25_000, False))


def test_a_candidate_that_crashes_is_not_treated_as_too_big(monkeypatch) -> None:
    """Only memory means "too big". A crash silently downgraded into a narrower
    launch would look exactly like a card that could not hold the batch."""

    class _Completed:
        returncode = 1
        stdout = json.dumps({"outcome": "error", "error": "ImportError: no such module"})
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *_, **__: _Completed())
    runner = vram_probe.subprocess_runner(profile="rl", device="cuda", compile_mode=None)
    with pytest.raises(VramError, match="other than memory: ImportError"):
        runner(VramKnobs(3904, 25_000, False))


def test_an_out_of_memory_candidate_is_a_rejection_not_a_failure(monkeypatch) -> None:
    class _Completed:
        returncode = OOM_EXIT_CODE
        stdout = json.dumps({"outcome": "oom", "error": "CUDA out of memory"})
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *_, **__: _Completed())
    runner = vram_probe.subprocess_runner(profile="rl", device="cuda", compile_mode=None)
    outcome = runner(VramKnobs(3904, 25_000, False))
    assert not outcome.fit
    assert outcome.reason == "oom"


def test_a_child_that_hangs_is_not_silently_accepted(monkeypatch) -> None:
    def _timeout(*_, **__):
        raise subprocess.TimeoutExpired(cmd="probe", timeout=1.0)

    monkeypatch.setattr(subprocess, "run", _timeout)
    runner = vram_probe.subprocess_runner(
        profile="rl", device="cuda", compile_mode=None, timeout=1.0
    )
    with pytest.raises(VramError, match="did not finish within"):
        runner(VramKnobs(3904, 25_000, False))


def test_the_child_reports_an_out_of_memory_candidate_distinctly() -> None:
    assert OOM_EXIT_CODE not in (0, 1), "OOM has to be distinguishable from a crash"


# ----------------------------------------------------------------------
# Cache lookup
# ----------------------------------------------------------------------


def test_auto_without_a_cache_keeps_the_profile_sizing(tmp_path: Path, fake_cuda) -> None:
    resolution = resolve_vram(
        VramPolicy("auto"),
        profile_name="rl",
        profile_fingerprint=_RL_FINGERPRINT,
        device="cuda",
        compile_mode=None,
        cache_file=tmp_path / ".vram.json",
    )
    assert resolution.status == "unresolved"
    assert resolution.applied.is_empty
    assert "run --vram probe" in resolution.notes[0]


def test_auto_uses_a_matching_measurement(tmp_path: Path, fake_cuda) -> None:
    cache = tmp_path / ".vram.json"
    entry, _ = probe_profile(
        "rl",
        device="cuda",
        compile_mode=None,
        profile_fingerprint=_RL_FINGERPRINT,
        runner=_runner(VramKnobs(1952, 25_000, True)),
        identity=_IDENTITY,
    )
    write_cache_entry(cache, entry)

    resolution = resolve_vram(
        VramPolicy("auto"),
        profile_name="rl",
        profile_fingerprint=_RL_FINGERPRINT,
        device="cuda",
        compile_mode=None,
        cache_file=cache,
    )
    assert resolution.status == "measured"
    assert resolution.source == "vram-cache"
    assert resolution.applied == VramKnobs(1952, 25_000, True)


@pytest.mark.parametrize(
    ("field", "value"),
    (("profile_fingerprint", "0" * 64), ("compile_mode", "max-autotune")),
)
def test_auto_refuses_a_measurement_that_answered_another_question(
    tmp_path: Path, fake_cuda, field: str, value
) -> None:
    cache = tmp_path / ".vram.json"
    entry, _ = probe_profile(
        "rl",
        device="cuda",
        compile_mode=None,
        profile_fingerprint=_RL_FINGERPRINT,
        runner=_runner(VramKnobs(1952, 25_000, True)),
        identity=_IDENTITY,
    )
    write_cache_entry(cache, entry)

    arguments = {
        "profile_name": "rl",
        "profile_fingerprint": _RL_FINGERPRINT,
        "device": "cuda",
        "compile_mode": None,
        field: value,
    }
    resolution = resolve_vram(VramPolicy("auto"), cache_file=cache, **arguments)
    assert resolution.status == "unresolved"
    assert "none matching this machine" in resolution.notes[0]


def test_probe_reuses_a_stored_measurement_but_reprobe_replaces_it(
    tmp_path: Path, fake_cuda
) -> None:
    cache = tmp_path / ".vram.json"
    first = _runner(VramKnobs(3904, 25_000, False))
    resolve_vram(
        VramPolicy("probe"),
        profile_name="rl",
        profile_fingerprint=_RL_FINGERPRINT,
        device="cuda",
        compile_mode=None,
        cache_file=cache,
        runner=first,
    )
    assert first.attempted == [VramKnobs(3904, 25_000, False)]
    assert len(read_cache(cache)) == 1

    idle = _runner(VramKnobs(3904, 25_000, False))
    reused = resolve_vram(
        VramPolicy("probe"),
        profile_name="rl",
        profile_fingerprint=_RL_FINGERPRINT,
        device="cuda",
        compile_mode=None,
        cache_file=cache,
        runner=idle,
    )
    assert idle.attempted == [], "probe must not re-measure what it already knows"
    assert reused.status == "measured"

    again = _runner(VramKnobs(3904, 25_000, True))
    replaced = resolve_vram(
        VramPolicy("reprobe"),
        profile_name="rl",
        profile_fingerprint=_RL_FINGERPRINT,
        device="cuda",
        compile_mode=None,
        cache_file=cache,
        runner=again,
    )
    assert again.attempted == [VramKnobs(3904, 25_000, False), VramKnobs(3904, 25_000, True)]
    assert replaced.applied == VramKnobs(3904, 25_000, True)
    assert len(read_cache(cache)) == 1, "the same machine keeps one entry"


def test_probing_writes_the_cache_and_reports_what_it_rejected(tmp_path: Path, fake_cuda) -> None:
    cache = tmp_path / ".vram.json"
    lines: list[str] = []
    resolve_vram(
        VramPolicy("probe"),
        profile_name="rl",
        profile_fingerprint=_RL_FINGERPRINT,
        device="cuda",
        compile_mode=None,
        cache_file=cache,
        runner=_runner(VramKnobs(3904, 25_000, True)),
        report=lines.append,
    )
    assert any("probing VRAM" in line for line in lines)
    assert any("rejected" in line for line in lines)
    assert any(str(cache) in line for line in lines)
    assert cache.is_file()


def test_off_and_a_non_accelerator_never_look_at_a_device(tmp_path: Path, monkeypatch) -> None:
    def _forbidden(_device):
        raise AssertionError("the device must not be queried")

    monkeypatch.setattr(vram_probe, "device_identity", _forbidden)
    cache = tmp_path / ".vram.json"
    cache.write_text("this would raise if it were read")

    off = resolve_vram(
        VramPolicy("off"),
        profile_name="rl",
        profile_fingerprint=_RL_FINGERPRINT,
        device="cuda",
        compile_mode=None,
        cache_file=cache,
    )
    assert off.status == "unresolved" and "--vram off" in off.notes[0]

    on_cpu = resolve_vram(
        VramPolicy("auto"),
        profile_name="rl",
        profile_fingerprint=_RL_FINGERPRINT,
        device="cpu",
        compile_mode=None,
        cache_file=cache,
    )
    assert on_cpu.status == "unresolved" and "not an accelerator" in on_cpu.notes[0]


def test_probing_without_an_accelerator_fails_loudly(monkeypatch) -> None:
    with pytest.raises(VramError, match="needs a CUDA device"):
        resolve_vram(
            VramPolicy("probe"),
            profile_name="rl",
            profile_fingerprint=_RL_FINGERPRINT,
            device="cpu",
            compile_mode=None,
        )


def test_a_preset_needs_no_device_at_all(monkeypatch) -> None:
    monkeypatch.setattr(
        vram_probe,
        "device_identity",
        lambda _device: (_ for _ in ()).throw(AssertionError("queried")),
    )
    resolution = resolve_vram(
        VramPolicy("preset", 16),
        profile_name="rl",
        profile_fingerprint=_RL_FINGERPRINT,
        device="cpu",
        compile_mode=None,
    )
    assert resolution.applied == VramKnobs(5856, 37_500, False)
    assert resolution.status == "provisional"
    # `auto` calls a CPU launch "nothing to size" and a preset sizes it anyway;
    # the record says which happened rather than leaving the two to disagree.
    assert any("not an accelerator" in note for note in resolution.notes)
    assert any("because it was asked for" in note for note in resolution.notes)


# ----------------------------------------------------------------------
# Complete launch composition
# ----------------------------------------------------------------------


def test_a_default_cpu_launch_resolves_exactly_as_the_profile_does() -> None:
    launch = resolve_training_launch(profile="rl", device="cpu")
    baseline = resolve_profile(PROFILES["rl"])

    assert launch.resolved.resolved_config_fingerprint == baseline.resolved_config_fingerprint
    assert launch.vram.status == "unresolved"
    assert launch.document()["vram"]["tiers"] == {}
    assert launch.resolved.value_sources["train_config.scales.0.num_envs"] == "derived"


def test_the_shipped_row_records_no_tier_because_it_moves_nothing() -> None:
    """`--vram 8` is the shipped launch down to the fingerprint, so the record
    must not warn about tier 2's different minibatch composition."""

    launch = resolve_training_launch(profile="rl", vram="8", device="cpu")
    baseline = resolve_profile(PROFILES["rl"])
    record = launch.document()["vram"]

    assert launch.resolved.resolved_config_fingerprint == baseline.resolved_config_fingerprint
    assert record["applied"] == launch.baseline.document()
    assert record["tiers"] == {}
    # The proposal is still recorded in full; only the claim about it changed.
    assert record["proposed"]["num_envs"] == 3904


def test_a_measured_launch_records_every_source(tmp_path: Path, fake_cuda) -> None:
    cache = tmp_path / ".vram.json"
    entry, _ = probe_profile(
        "rl",
        device="cuda",
        compile_mode=None,
        profile_fingerprint=_RL_FINGERPRINT,
        runner=_runner(VramKnobs(1952, 25_000, True)),
        identity=_IDENTITY,
    )
    write_cache_entry(cache, entry)

    launch = resolve_training_launch(
        profile="rl", vram="auto", device="cuda", compile_mode=None, cache_file=cache
    )
    sources = launch.resolved.value_sources
    assert sources["train_config.scales.0.num_envs"] == "vram-cache"
    assert sources["train_config.microbatch_tokens"] == "vram-cache"
    assert sources["model_config.grad_checkpoint"] == "vram-cache"
    assert launch.resolved.train_config.scales[0].num_envs == 1952
    assert launch.resolved.train_config.rollouts_per_update == 6
    assert launch.resolved.model_config.grad_checkpoint is True

    record = launch.document()["vram"]
    assert record["status"] == "measured"
    assert record["identity_fingerprint"] == entry.fingerprint
    assert set(record["tiers"]) == {"1", "2"}


def test_the_vram_decision_is_stored_in_the_checkpoint(tmp_path: Path, fake_cuda) -> None:
    """A run's memory decision belongs to its history, not to the machine that
    happened to start it."""

    import torch

    from boost_and_broadside.train.rl.checkpoint import (
        build_policy_checkpoint_payload,
        write_checkpoint_payload,
    )

    cache = tmp_path / ".vram.json"
    entry, _ = probe_profile(
        "rl",
        device="cuda",
        compile_mode=None,
        profile_fingerprint=_RL_FINGERPRINT,
        runner=_runner(VramKnobs(1952, 25_000, True)),
        identity=_IDENTITY,
    )
    write_cache_entry(cache, entry)
    launch = resolve_training_launch(
        profile="rl", vram="auto", device="cuda", compile_mode=None, cache_file=cache
    )

    payload = build_policy_checkpoint_payload(
        policy_state_dict={},
        num_value_components=1,
        team_pma_k=(1,),
        global_step=0,
        live_elo=0.0,
        model_config=launch.resolved.model_config,
        env_config=launch.resolved.env_config,
        ship_config=launch.resolved.ship_config,
        paradigm=launch.resolved.train_config.paradigm,
        launch=launch.document(),
    )
    path = write_checkpoint_payload(tmp_path / "step_0.pt", payload)
    stored = torch.load(path, weights_only=False)["launch"]["vram"]

    assert stored["status"] == "measured"
    assert stored["source"] == "vram-cache"
    assert stored["applied"] == {
        "num_envs": 1952,
        "microbatch_tokens": 25_000,
        "grad_checkpoint": True,
    }
    assert stored["identity_fingerprint"] == entry.fingerprint
    assert payload["model_config"]["grad_checkpoint"] is True


def test_an_explicit_override_outranks_a_measurement(tmp_path: Path, fake_cuda) -> None:
    cache = tmp_path / ".vram.json"
    entry, _ = probe_profile(
        "rl",
        device="cuda",
        compile_mode=None,
        profile_fingerprint=_RL_FINGERPRINT,
        runner=_runner(VramKnobs(1952, 25_000, True)),
        identity=_IDENTITY,
    )
    write_cache_entry(cache, entry)

    launch = resolve_training_launch(
        profile="rl",
        vram="auto",
        device="cuda",
        compile_mode=None,
        cache_file=cache,
        num_envs=5856,
    )
    assert launch.resolved.train_config.scales[0].num_envs == 5856
    assert launch.resolved.value_sources["train_config.scales.0.num_envs"] == "cli"
    # The measurement still sized the knobs the command line did not name.
    assert launch.resolved.value_sources["train_config.microbatch_tokens"] == "vram-cache"
    assert any("overrides the vram-cache width" in note for note in launch.vram.notes)


def test_an_override_that_would_resize_the_logical_batch_still_fails(fake_cuda) -> None:
    with pytest.raises(ValueError, match="fixed logical batch"):
        resolve_training_launch(profile="rl", vram="off", device="cpu", num_envs=3872)


def test_print_config_refuses_to_probe() -> None:
    with pytest.raises(UserFacingError, match="--print-config cannot run --vram probe"):
        resolve_training_launch(profile="rl", vram="probe", device="cpu", allow_probe=False)


@pytest.mark.parametrize("pin", ({"num_envs": 1952}, {"microbatch_tokens": 25_000}))
def test_a_probe_cannot_be_asked_to_measure_a_pinned_knob(pin: dict) -> None:
    with pytest.raises(UserFacingError, match="determines --num-envs and --microbatch-tokens"):
        resolve_training_launch(profile="rl", vram="reprobe", device="cuda", **pin)


def test_an_invalid_policy_is_rejected_before_anything_is_resolved() -> None:
    with pytest.raises(VramError, match="invalid --vram value"):
        resolve_training_launch(profile="rl", vram="12", device="cpu")
