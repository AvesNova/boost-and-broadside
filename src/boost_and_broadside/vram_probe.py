"""Measure which VRAM launch actually fits, one fresh subprocess at a time.

The parent process never allocates a candidate.  Each configuration is measured
in its own interpreter, so an out-of-memory failure cannot leave a fragmented
allocator behind for the next attempt, and the first candidate that survives a
complete update is the answer.

Run as a module, this file *is* the child::

    python -m boost_and_broadside.vram_probe --profile rl --device cuda \\
        --num-envs 3904 --microbatch-tokens 25000

It prints one JSON object describing the outcome and exits 0 when the
configuration fit, :data:`OOM_EXIT_CODE` when it ran out of memory, and 1 when
anything else went wrong.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from boost_and_broadside.config.resolve import LaunchGeometry, LaunchOverrides, launch_geometry
from boost_and_broadside.config.vram import (
    PROBE_VERSION,
    VRAM_PRESETS,
    VramCacheEntry,
    VramError,
    VramKnobs,
    VramPolicy,
    VramResolution,
    cache_identity,
    cache_path,
    identity_fingerprint,
    preset_knobs,
    read_cache,
    resolution_from_cache,
    resolution_from_preset,
    unresolved,
    utc_timestamp,
    write_cache_entry,
)
from boost_and_broadside.profiles import PROFILES

OOM_EXIT_CODE = 3

# One candidate has to collect a full logical batch and run one complete PPO
# update at production width. Generous, but not unbounded.
DEFAULT_CANDIDATE_TIMEOUT_SECONDS = 3600.0

_GIB = 1024**3
_PRESET_MATCH_FRACTION = 0.9


def device_identity(device: str) -> dict[str, Any]:
    """Identify the accelerator a measurement is only valid for.

    MIG instances are reported separately: they expose a slice of a physical
    card under its own UUID, so a cache entry measured on one must never be
    reused on the whole card or on a differently sized slice.
    """

    if not device.startswith("cuda"):
        raise VramError(f"VRAM probing needs a CUDA device, got --device {device}")
    if not torch.cuda.is_available():
        raise VramError("VRAM probing needs a CUDA device, but CUDA is unavailable")
    index = torch.device(device).index or 0
    properties = torch.cuda.get_device_properties(index)
    uuid = getattr(properties, "uuid", None)
    uuid_text = None if uuid is None else str(uuid)
    return {
        "name": properties.name,
        "uuid": uuid_text,
        "mig": bool(uuid_text and uuid_text.startswith("MIG")) or "MIG" in properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "capability": f"{properties.major}.{properties.minor}",
        "multi_processor_count": int(properties.multi_processor_count),
    }


def software_identity() -> dict[str, Any]:
    """The software stack a measurement is only valid for."""

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }


def candidate_knobs(geometry: LaunchGeometry, *, total_memory_bytes: int) -> tuple[VramKnobs, ...]:
    """The ordered ladder a probe walks, most desirable configuration first.

    It starts from the preset rows the project actually recommends, skipping
    rows that target more memory than the device advertises, and then falls back
    below the smallest row: gradient checkpointing first, because it is exact
    and only costs time, and a narrower rollout shard only after that.
    """

    ladder: list[VramKnobs] = []

    def add(knobs: VramKnobs) -> None:
        if knobs not in ladder:
            ladder.append(knobs)

    # Preset rows are named for a card's marketed capacity, which is always a
    # little more than the memory CUDA reports (an "8 GB" card advertises about
    # 7.6 GiB), so a row matches slightly below its nominal size.
    advertised_gigabytes = total_memory_bytes / _GIB
    rows = [
        gigabytes
        for gigabytes in sorted(VRAM_PRESETS, reverse=True)
        if gigabytes * _PRESET_MATCH_FRACTION <= advertised_gigabytes
    ] or [min(VRAM_PRESETS)]
    for gigabytes in rows:
        add(preset_knobs(VRAM_PRESETS[gigabytes], geometry))

    narrowest = ladder[-1]
    add(replace(narrowest, grad_checkpoint=True))
    assert narrowest.num_envs is not None
    narrower = [
        num_envs for num_envs, _shards in geometry.shard_widths() if num_envs < narrowest.num_envs
    ]
    for num_envs in narrower[:2]:
        microbatch = min(
            narrowest.microbatch_tokens or geometry.minibatch_tokens(num_envs),
            geometry.minibatch_tokens(num_envs),
        )
        add(VramKnobs(num_envs=num_envs, microbatch_tokens=microbatch, grad_checkpoint=True))
    return tuple(ladder)


@dataclass(frozen=True)
class ProbeOutcome:
    """The result of measuring exactly one candidate in its own process."""

    knobs: VramKnobs
    fit: bool
    measurement: Mapping[str, Any]

    @property
    def reason(self) -> str:
        return str(self.measurement.get("outcome", "unknown"))


CandidateRunner = Callable[[VramKnobs], ProbeOutcome]


def _child_command(
    knobs: VramKnobs,
    *,
    profile: str,
    device: str,
    compile_mode: str | None,
) -> list[str]:
    assert knobs.num_envs is not None and knobs.microbatch_tokens is not None
    command = [
        sys.executable,
        "-m",
        "boost_and_broadside.vram_probe",
        "--profile",
        profile,
        "--device",
        device,
        "--num-envs",
        str(knobs.num_envs),
        "--microbatch-tokens",
        str(knobs.microbatch_tokens),
        "--compile",
        compile_mode or "none",
    ]
    if knobs.grad_checkpoint:
        command.append("--grad-checkpoint")
    return command


def subprocess_runner(
    *,
    profile: str,
    device: str,
    compile_mode: str | None,
    timeout: float = DEFAULT_CANDIDATE_TIMEOUT_SECONDS,
) -> CandidateRunner:
    """Measure candidates in fresh interpreters rooted in a scratch directory."""

    def run(knobs: VramKnobs) -> ProbeOutcome:
        command = _child_command(
            knobs, profile=profile, device=device, compile_mode=compile_mode
        )
        with tempfile.TemporaryDirectory(prefix="bnb-vram-probe-") as scratch:
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=scratch,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                raise VramError(
                    f"VRAM probe candidate {knobs.document()} did not finish within "
                    f"{timeout:.0f}s"
                ) from error
        measurement = _parse_child_output(completed.stdout)
        if measurement is None:
            tail = (completed.stderr or "").strip().splitlines()[-5:]
            raise VramError(
                f"VRAM probe candidate {knobs.document()} failed with exit code "
                f"{completed.returncode} and no result: " + " | ".join(tail)
            )
        return ProbeOutcome(
            knobs=knobs,
            fit=completed.returncode == 0 and measurement.get("outcome") == "fit",
            measurement=measurement,
        )

    return run


def _parse_child_output(stdout: str) -> dict[str, Any] | None:
    for line in reversed((stdout or "").strip().splitlines()):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "outcome" in parsed:
            return parsed
    return None


def probe_profile(
    profile_name: str,
    *,
    device: str,
    compile_mode: str | None,
    profile_fingerprint: str,
    runner: CandidateRunner | None = None,
    identity: Mapping[str, Any] | None = None,
) -> tuple[VramCacheEntry, tuple[ProbeOutcome, ...]]:
    """Walk the candidate ladder and return the first configuration that fits.

    Returns the cache entry to store along with every attempt, so a caller can
    report the rejected candidates instead of hiding them.
    """

    if profile_name not in PROFILES:
        raise VramError(f"unknown profile {profile_name!r}")
    geometry = launch_geometry(PROFILES[profile_name])
    device_record = dict(identity) if identity is not None else device_identity(device)
    full_identity = cache_identity(
        profile_name=profile_name,
        profile_fingerprint=profile_fingerprint,
        geometry=geometry,
        compile_mode=compile_mode,
        device=device_record,
        software=software_identity(),
    )
    run = runner or subprocess_runner(
        profile=profile_name, device=device, compile_mode=compile_mode
    )
    attempts: list[ProbeOutcome] = []
    for knobs in candidate_knobs(
        geometry, total_memory_bytes=int(device_record["total_memory_bytes"])
    ):
        outcome = run(knobs)
        attempts.append(outcome)
        if outcome.fit:
            entry = VramCacheEntry(
                fingerprint=identity_fingerprint(full_identity),
                created=utc_timestamp(),
                identity=full_identity,
                knobs=knobs,
                measurement={
                    **dict(outcome.measurement),
                    "candidates_tried": len(attempts),
                    "rejected": [
                        {"knobs": attempt.knobs.document(), "outcome": attempt.reason}
                        for attempt in attempts[:-1]
                    ],
                },
            )
            return entry, tuple(attempts)
    rejected = ", ".join(f"{attempt.knobs.document()} ({attempt.reason})" for attempt in attempts)
    raise VramError(
        f"no probed configuration fit on {device_record.get('name', device)}: {rejected}"
    )


def resolve_vram(
    policy: VramPolicy,
    *,
    profile_name: str,
    profile_fingerprint: str,
    device: str,
    compile_mode: str | None,
    cache_file: Path | None = None,
    runner: CandidateRunner | None = None,
    report: Callable[[str], None] | None = None,
) -> VramResolution:
    """Turn a ``--vram`` policy into a launch proposal, probing when asked.

    Nothing here touches the accelerator unless the policy actually needs it:
    ``--vram off``, a non-CUDA device, and ``--vram auto`` with no cache file
    all return without querying a device, so the common launch resolves exactly
    as it did before this system existed.
    """

    announce = report or (lambda _message: None)
    if policy.mode == "off":
        return unresolved(policy, "--vram off: the profile's own derived sizing is used")
    if policy.mode == "preset":
        assert policy.preset_gigabytes is not None
        geometry = launch_geometry(PROFILES[profile_name])
        return resolution_from_preset(policy, VRAM_PRESETS[policy.preset_gigabytes], geometry)

    if not device.startswith("cuda"):
        if policy.probes:
            raise VramError(f"--vram {policy} needs a CUDA device, got --device {device}")
        return unresolved(policy, f"--device {device} is not an accelerator; nothing to size")

    path = cache_file or cache_path()
    if policy.mode == "auto" and not path.is_file():
        return unresolved(
            policy,
            f"no VRAM cache at {path}; run --vram probe to measure this machine, "
            "or --vram 8|16|24|32 for a provisional preset",
        )

    geometry = launch_geometry(PROFILES[profile_name])
    device_record = device_identity(device)
    identity = cache_identity(
        profile_name=profile_name,
        profile_fingerprint=profile_fingerprint,
        geometry=geometry,
        compile_mode=compile_mode,
        device=device_record,
        software=software_identity(),
    )
    key = identity_fingerprint(identity)
    if policy.reads_cache:
        entries = read_cache(path)
        entry = entries.get(key)
        if entry is not None:
            return resolution_from_cache(policy, entry)
        if policy.mode == "auto":
            return unresolved(
                policy,
                f"{path} has {len(entries)} entr{'y' if len(entries) == 1 else 'ies'}, none "
                f"matching this machine and profile ({key[:12]}); run --vram probe",
            )

    announce(f"probing VRAM for profile {profile_name!r} on {device}; this runs real updates")
    entry, attempts = probe_profile(
        profile_name,
        device=device,
        compile_mode=compile_mode,
        profile_fingerprint=profile_fingerprint,
        runner=runner,
        # The device was already identified above; asking the driver twice
        # cannot improve the answer and could only disagree with it.
        identity=device_record,
    )
    for attempt in attempts[:-1]:
        announce(f"  rejected {attempt.knobs.document()}: {attempt.reason}")
    write_cache_entry(path, entry)
    announce(f"  measured {entry.knobs.document()}; cached in {path}")
    return replace(
        resolution_from_cache(policy, entry),
        notes=(
            f"measured by probe version {PROBE_VERSION} on {entry.created}",
            f"{len(attempts) - 1} candidate(s) rejected before this one",
        ),
    )


# ----------------------------------------------------------------------
# Child process: measure exactly one candidate
# ----------------------------------------------------------------------


def measure_candidate(
    profile_name: str,
    knobs: VramKnobs,
    *,
    device: str,
    compile_mode: str | None,
) -> dict[str, Any]:
    """Run one complete PPO update at this sizing and report the peak.

    Imported lazily by the child only, because it pulls in the trainer.
    """

    from dataclasses import replace as dataclass_replace

    from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
    from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
    from boost_and_broadside.profiles import resolve_named_profile
    from boost_and_broadside.train.rl.ppo import PPOTrainer

    resolved = resolve_named_profile(
        profile_name,
        LaunchOverrides(
            num_envs=knobs.num_envs,
            microbatch_tokens=knobs.microbatch_tokens,
            grad_checkpoint=knobs.grad_checkpoint,
            num_envs_source="vram-preset",
            microbatch_tokens_source="vram-preset",
            grad_checkpoint_source="vram-preset",
        ),
    )
    train_config = resolved.train_config
    # Exactly one update: `_num_updates` is the timestep budget divided by the
    # env-steps one update consumes, so this stops the loop after the first.
    one_update = (
        sum(scale.num_envs for scale in train_config.scales)
        * train_config.num_steps
        * train_config.rollouts_per_update
    )
    torch.cuda.reset_peak_memory_stats()
    trainer = PPOTrainer(
        train_config=dataclass_replace(train_config, total_timesteps=one_update),
        model_config=resolved.model_config,
        ship_config=resolved.ship_config,
        device=device,
        use_wandb=False,
        scripted_agent=StochasticScriptedAgent(resolved.ship_config, StochasticAgentConfig()),
        compile_mode=compile_mode,
    )
    try:
        trainer.train()
    finally:
        trainer.shutdown()
    index = torch.device(device).index or 0
    return {
        "outcome": "fit",
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
        "total_memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
        "knobs": knobs.document(),
    }


def _child_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m boost_and_broadside.vram_probe",
        description="Measure one VRAM candidate in this process and print the result.",
    )
    parser.add_argument("--profile", choices=tuple(sorted(PROFILES)), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--microbatch-tokens", type=int, required=True)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--compile", dest="compile_mode", default="none")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _child_parser().parse_args(argv)
    knobs = VramKnobs(
        num_envs=args.num_envs,
        microbatch_tokens=args.microbatch_tokens,
        grad_checkpoint=args.grad_checkpoint,
    )
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        print(
            json.dumps(
                {
                    "outcome": "error",
                    "error": f"VRAM probing needs a CUDA device, got --device {args.device}",
                    "knobs": knobs.document(),
                }
            )
        )
        return 1
    compile_mode = None if args.compile_mode == "none" else args.compile_mode
    try:
        result = measure_candidate(
            args.profile, knobs, device=args.device, compile_mode=compile_mode
        )
    except torch.cuda.OutOfMemoryError as error:
        print(
            json.dumps(
                {"outcome": "oom", "error": str(error)[:400], "knobs": knobs.document()}
            )
        )
        return OOM_EXIT_CODE
    except Exception as error:  # noqa: BLE001 - reported as JSON, never swallowed
        print(
            json.dumps(
                {
                    "outcome": "error",
                    "error": f"{type(error).__name__}: {error}"[:400],
                    "knobs": knobs.document(),
                }
            )
        )
        return 1
    print(json.dumps(result))
    return 0


__all__ = [
    "DEFAULT_CANDIDATE_TIMEOUT_SECONDS",
    "OOM_EXIT_CODE",
    "CandidateRunner",
    "ProbeOutcome",
    "candidate_knobs",
    "device_identity",
    "main",
    "measure_candidate",
    "probe_profile",
    "resolve_vram",
    "software_identity",
    "subprocess_runner",
]


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
