"""VRAM launch resolution: policies, presets, probe cache, and tier labels.

This module is deliberately free of Torch.  It owns what a VRAM decision *is* —
which knobs may move, what each knob's equivalence guarantee is, which cache
entry applies to a machine, and how a preset maps onto one profile's fixed token
arithmetic.  Measuring an actual device belongs to
:mod:`boost_and_broadside.vram_probe`, which supplies the device and software
identity this module fingerprints.

Three ideas carry the design:

* **Tiers.**  A knob is only usable here if it keeps the experiment intact.
  Tier 1 preserves the mathematical objective, tier 2 preserves the nominal
  logical batch, and tier 3 — total token budget, minibatch count, fleet size —
  is an experiment change that no VRAM decision may make.
* **Measured versus provisional.**  A result is ``measured`` only when it came
  from a probe of *this* machine, recorded in the cache under a fingerprint that
  still matches.  Everything else, including a preset row whose numbers were
  measured on somebody else's card, is ``provisional`` and says so.
* **Precedence.**  VRAM proposes; the command line disposes.  An explicit
  ``--num-envs``/``--microbatch-tokens`` always wins, and the source map records
  which knob came from where.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from boost_and_broadside.config.fingerprint import canonical_data, fingerprint
from boost_and_broadside.config.resolve import LaunchGeometry, LaunchOverrides
from boost_and_broadside.config.schema import ResolutionSource

VRAM_CACHE_FILENAME = ".vram.json"
VRAM_CACHE_SCHEMA_VERSION = 1

# Bump when a probe measures something different enough that older entries no
# longer describe the same question.  It is part of the cache fingerprint, so a
# bump invalidates every stored entry.
PROBE_VERSION = 1

# The trainer runs its forward/backward under bf16 autocast (see ``ppo.py``).
# Activation memory depends on it, so it is part of the cache identity.
AUTOCAST_DTYPE = "bfloat16"

TIER_GUARANTEES: Mapping[int, str] = {
    1: (
        "same mathematical objective: equivalent update within floating-point "
        "tolerance, not bit-identical"
    ),
    2: (
        "same nominal logical batch and optimizer-step count: different env-stream "
        "count, temporal correlation, and minibatch composition"
    ),
    3: "experiment change: alters the optimization or the task",
}

# Every knob VRAM resolution is allowed to move, and the tier it belongs to.
# ``num_envs`` is tier 2 because it redistributes a fixed logical batch across a
# different number of rollout shards; nothing here is tier 3.
KNOB_TIERS: Mapping[str, int] = {
    "grad_checkpoint": 1,
    "microbatch_tokens": 1,
    "num_envs": 2,
}


class VramError(ValueError):
    """A VRAM policy, preset, or cache problem the user has to resolve."""


@dataclass(frozen=True)
class VramPolicy:
    """A parsed ``--vram`` value."""

    mode: Literal["auto", "probe", "reprobe", "off", "preset"]
    preset_gigabytes: int | None = None

    def __str__(self) -> str:
        return str(self.preset_gigabytes) if self.mode == "preset" else self.mode

    @property
    def probes(self) -> bool:
        return self.mode in {"probe", "reprobe"}

    @property
    def reads_cache(self) -> bool:
        return self.mode in {"auto", "probe"}


@dataclass(frozen=True)
class VramKnobs:
    """Concrete launch values a VRAM decision proposes for one profile."""

    num_envs: int | None = None
    microbatch_tokens: int | None = None
    grad_checkpoint: bool | None = None

    def document(self) -> dict[str, Any]:
        return {
            "num_envs": self.num_envs,
            "microbatch_tokens": self.microbatch_tokens,
            "grad_checkpoint": self.grad_checkpoint,
        }

    @property
    def is_empty(self) -> bool:
        return self.document() == VramKnobs().document()

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> VramKnobs:
        unknown = set(document) - set(VramKnobs().document())
        if unknown:
            raise VramError(f"unknown VRAM knobs: {sorted(unknown)}")
        return cls(
            num_envs=document.get("num_envs"),
            microbatch_tokens=document.get("microbatch_tokens"),
            grad_checkpoint=document.get("grad_checkpoint"),
        )

    def tiers(self) -> tuple[int, ...]:
        """The equivalence tiers this proposal actually touches."""

        moved = {name for name, value in self.document().items() if value is not None}
        return tuple(sorted({KNOB_TIERS[name] for name in moved}))


@dataclass(frozen=True)
class VramPreset:
    """One memory-tier row.

    ``max_rollout_tokens`` is the entity-token ceiling for a single resident
    rollout shard, which is what the persistent buffer and the rollout peak
    scale with.  It is stated as a ceiling rather than a width because the valid
    widths differ per profile: a shard has to divide the profile's fixed logical
    batch exactly and stay minibatch-aligned.

    ``measured_on`` names the device a row's numbers were taken on, or ``None``
    when the row is an extrapolation.  Either way, applying a preset to *this*
    machine is reported as provisional — only a probe of this machine measures
    it.
    """

    gigabytes: int
    max_rollout_tokens: int
    microbatch_tokens: int
    grad_checkpoint: bool
    measured_on: str | None
    basis: str


# The 8 GB row is exactly the launch every registered profile already resolves
# to, so `--vram 8` changes nothing on the card the project was developed on.
# The larger rows spend their headroom on tier 2 first (fewer, wider shards,
# which the memory notes show is where throughput is) and only then on tier 1.
VRAM_PRESETS: Mapping[int, VramPreset] = {
    preset.gigabytes: preset
    for preset in (
        VramPreset(
            gigabytes=8,
            max_rollout_tokens=4_000_000,
            microbatch_tokens=25_000,
            grad_checkpoint=False,
            measured_on="NVIDIA GeForce RTX 4070 Laptop GPU",
            basis=(
                "the shipped launch preset; the Jul 2026 sweep in "
                "docs/engineering/memory-optimization.md found the microbatch divisor "
                "of 5 to be the largest that fits this card"
            ),
        ),
        VramPreset(
            gigabytes=16,
            max_rollout_tokens=6_000_000,
            microbatch_tokens=37_500,
            grad_checkpoint=False,
            measured_on=None,
            basis=(
                "provisional: linear extrapolation of the 8 GB persistent/rollout "
                "peaks to a ~6M-token shard, with the microbatch held at the value "
                "the production comparison measured"
            ),
        ),
        VramPreset(
            gigabytes=24,
            max_rollout_tokens=12_000_000,
            microbatch_tokens=37_500,
            grad_checkpoint=False,
            measured_on=None,
            basis=(
                "provisional: the whole logical batch resident in one shard, with "
                "the microbatch held at the measured production value"
            ),
        ),
        VramPreset(
            gigabytes=32,
            max_rollout_tokens=12_000_000,
            microbatch_tokens=75_000,
            grad_checkpoint=False,
            measured_on=None,
            basis=(
                "provisional: one resident shard as at 24 GB, spending the remaining "
                "headroom on a doubled microbatch; never measured"
            ),
        ),
    )
}


def parse_vram_policy(value: str) -> VramPolicy:
    """Parse ``auto|probe|reprobe|off|<gigabytes>``."""

    if value in {"auto", "probe", "reprobe", "off"}:
        return VramPolicy(value)  # type: ignore[arg-type]
    if value.isdigit() and int(value) in VRAM_PRESETS:
        return VramPolicy("preset", int(value))
    choices = "|".join(["auto", "probe", "reprobe", "off", *(str(g) for g in sorted(VRAM_PRESETS))])
    raise VramError(f"invalid --vram value {value!r}; expected {choices}")


def preset_knobs(preset: VramPreset, geometry: LaunchGeometry) -> VramKnobs:
    """Map a memory-tier row onto one profile's valid shard widths.

    Selects the widest shard that stays within the row's ceiling, so a bigger
    card holds more of the fixed logical batch resident at once.
    """

    candidates = [
        (num_envs, shards)
        for num_envs, shards in geometry.shard_widths()
        if geometry.rollout_tokens(num_envs) <= preset.max_rollout_tokens
    ]
    if not candidates:
        raise VramError(
            f"no shard width preserves the fixed logical batch of "
            f"{geometry.aligned_logical_batch_tokens} entity tokens within the "
            f"{preset.gigabytes} GB ceiling of {preset.max_rollout_tokens} tokens per shard"
        )
    num_envs, _shards = candidates[0]
    microbatch_tokens = min(preset.microbatch_tokens, geometry.minibatch_tokens(num_envs))
    return VramKnobs(
        num_envs=num_envs,
        microbatch_tokens=microbatch_tokens,
        grad_checkpoint=preset.grad_checkpoint,
    )


def cache_identity(
    *,
    profile_name: str,
    profile_fingerprint: str,
    geometry: LaunchGeometry,
    compile_mode: str | None,
    device: Mapping[str, Any],
    software: Mapping[str, Any],
) -> dict[str, Any]:
    """The complete description a cached measurement is only valid for.

    ``profile_fingerprint`` already binds the model and the rollout intent; the
    explicit token arithmetic is repeated so ``.vram.json`` stays readable by a
    human deciding whether an entry is still relevant.
    """

    return {
        "probe_version": PROBE_VERSION,
        "autocast_dtype": AUTOCAST_DTYPE,
        "compile_mode": compile_mode,
        "profile": profile_name,
        "profile_fingerprint": profile_fingerprint,
        "geometry": {
            "entity_tokens": geometry.entity_tokens,
            "num_steps": geometry.num_steps,
            "num_minibatches": geometry.num_minibatches,
            "aligned_logical_batch_tokens": geometry.aligned_logical_batch_tokens,
        },
        "device": canonical_data(device),
        "software": canonical_data(software),
    }


def identity_fingerprint(identity: Mapping[str, Any]) -> str:
    return fingerprint(dict(identity))


@dataclass(frozen=True)
class VramCacheEntry:
    """One measured result, valid only for its exact identity fingerprint."""

    fingerprint: str
    created: str
    identity: Mapping[str, Any]
    knobs: VramKnobs
    measurement: Mapping[str, Any]

    def document(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "created": self.created,
            "identity": dict(self.identity),
            "knobs": self.knobs.document(),
            "measurement": dict(self.measurement),
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> VramCacheEntry:
        try:
            return cls(
                fingerprint=str(document["fingerprint"]),
                created=str(document["created"]),
                identity=dict(document["identity"]),
                knobs=VramKnobs.from_document(document["knobs"]),
                measurement=dict(document["measurement"]),
            )
        except (KeyError, TypeError) as error:
            raise VramError(f"malformed VRAM cache entry: {error}") from error


def cache_path(root: Path | str | None = None) -> Path:
    """Where the recomputable machine cache lives, relative to the working tree."""

    return Path(root or Path.cwd()) / VRAM_CACHE_FILENAME


def read_cache(path: Path) -> dict[str, VramCacheEntry]:
    """Read every stored entry, keyed by identity fingerprint.

    A missing file is simply an empty cache.  A file that exists but cannot be
    understood is an error rather than an empty cache: silently training at a
    different width because a recomputable file was damaged is exactly the kind
    of substitution this system exists to prevent.
    """

    if not path.is_file():
        return {}
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise VramError(
            f"{path} could not be read ({error}); delete it, or launch with "
            "--vram reprobe or --vram off"
        ) from error
    version = document.get("schema_version") if isinstance(document, dict) else None
    if version != VRAM_CACHE_SCHEMA_VERSION:
        raise VramError(
            f"{path} has schema version {version!r}, expected "
            f"{VRAM_CACHE_SCHEMA_VERSION}; delete it, or launch with "
            "--vram reprobe or --vram off"
        )
    entries = document.get("entries")
    if not isinstance(entries, list):
        raise VramError(f"{path} has no entry list; delete it, or launch with --vram reprobe")
    return {entry.fingerprint: entry for entry in map(VramCacheEntry.from_document, entries)}


def write_cache_entry(path: Path, entry: VramCacheEntry) -> None:
    """Replace one entry and rewrite the whole cache atomically."""

    entries = dict(read_cache(path)) if path.is_file() else {}
    entries[entry.fingerprint] = entry
    document = {
        "schema_version": VRAM_CACHE_SCHEMA_VERSION,
        "entries": [entries[key].document() for key in sorted(entries)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


@dataclass(frozen=True)
class VramResolution:
    """What a VRAM policy decided, and how honestly it knows it."""

    policy: str
    source: ResolutionSource | None
    status: Literal["measured", "provisional", "unresolved"]
    knobs: VramKnobs
    applied: VramKnobs
    identity_fingerprint: str | None
    notes: tuple[str, ...] = ()

    def document(self) -> dict[str, Any]:
        applied = self.applied.document()
        return {
            "policy": self.policy,
            "source": self.source,
            "status": self.status,
            "proposed": self.knobs.document(),
            "applied": applied,
            "tiers": {
                str(tier): TIER_GUARANTEES[tier] for tier in self.applied.tiers()
            },
            "identity_fingerprint": self.identity_fingerprint,
            "notes": list(self.notes),
        }


def unresolved(policy: VramPolicy, *notes: str) -> VramResolution:
    """A launch that keeps the profile's own derived sizing."""

    return VramResolution(
        policy=str(policy),
        source=None,
        status="unresolved",
        knobs=VramKnobs(),
        applied=VramKnobs(),
        identity_fingerprint=None,
        notes=notes,
    )


def resolution_from_cache(policy: VramPolicy, entry: VramCacheEntry) -> VramResolution:
    return VramResolution(
        policy=str(policy),
        source="vram-cache",
        status="measured",
        knobs=entry.knobs,
        applied=entry.knobs,
        identity_fingerprint=entry.fingerprint,
        notes=(f"measured by probe version {PROBE_VERSION} on {entry.created}",),
    )


def resolution_from_preset(
    policy: VramPolicy,
    preset: VramPreset,
    geometry: LaunchGeometry,
) -> VramResolution:
    knobs = preset_knobs(preset, geometry)
    origin = (
        f"row measured on {preset.measured_on}"
        if preset.measured_on
        else "row was never measured"
    )
    return VramResolution(
        policy=str(policy),
        source="vram-preset",
        status="provisional",
        knobs=knobs,
        applied=knobs,
        identity_fingerprint=None,
        notes=(
            f"{preset.gigabytes} GB preset ({origin}); {preset.basis}",
            "provisional for this machine: run --vram probe to measure it",
        ),
    )


def apply_cli_overrides(
    resolution: VramResolution,
    *,
    num_envs: int | None,
    microbatch_tokens: int | None,
) -> VramResolution:
    """Let explicit launch overrides win, and say which proposals they replaced."""

    applied = resolution.knobs
    notes = list(resolution.notes)
    if num_envs is not None:
        if applied.num_envs is not None and applied.num_envs != num_envs:
            notes.append(
                f"--num-envs {num_envs} overrides the {resolution.source} width {applied.num_envs}"
            )
        applied = replace(applied, num_envs=None)
    if microbatch_tokens is not None:
        if applied.microbatch_tokens is not None and applied.microbatch_tokens != microbatch_tokens:
            notes.append(
                f"--microbatch-tokens {microbatch_tokens} overrides the "
                f"{resolution.source} value {applied.microbatch_tokens}"
            )
        applied = replace(applied, microbatch_tokens=None)
    return replace(resolution, applied=applied, notes=tuple(notes))


def launch_overrides(
    resolution: VramResolution,
    *,
    num_envs: int | None,
    microbatch_tokens: int | None,
) -> LaunchOverrides:
    """Compose VRAM proposals and explicit CLI values into one override set.

    Precedence is the plan's: profile intent and derivations first, then the
    VRAM cache entry or preset, then explicit launch overrides, then validation
    inside :func:`~boost_and_broadside.config.resolve.resolve_profile`.
    """

    source: ResolutionSource = resolution.source or "cli"
    applied = resolution.applied
    return LaunchOverrides(
        num_envs=num_envs if num_envs is not None else applied.num_envs,
        microbatch_tokens=(
            microbatch_tokens if microbatch_tokens is not None else applied.microbatch_tokens
        ),
        grad_checkpoint=applied.grad_checkpoint,
        num_envs_source="cli" if num_envs is not None else source,
        microbatch_tokens_source="cli" if microbatch_tokens is not None else source,
        grad_checkpoint_source=source,
    )


__all__ = [
    "AUTOCAST_DTYPE",
    "KNOB_TIERS",
    "PROBE_VERSION",
    "TIER_GUARANTEES",
    "VRAM_CACHE_FILENAME",
    "VRAM_CACHE_SCHEMA_VERSION",
    "VRAM_PRESETS",
    "VramCacheEntry",
    "VramError",
    "VramKnobs",
    "VramPolicy",
    "VramPreset",
    "VramResolution",
    "apply_cli_overrides",
    "cache_identity",
    "cache_path",
    "identity_fingerprint",
    "launch_overrides",
    "parse_vram_policy",
    "preset_knobs",
    "read_cache",
    "resolution_from_cache",
    "resolution_from_preset",
    "unresolved",
    "utc_timestamp",
    "write_cache_entry",
]
