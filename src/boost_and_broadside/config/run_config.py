"""``checkpoints/<run>/config.json`` — what a run trained under, as data.

A run's configuration is not a value; it is a function of step. ``--continue``
extends an existing run with changed settings, so asking "what was this trained
with" has no answer until you say *when*. The file is therefore an append-only
list of segments, each recording the ``global_step`` it took effect at.

Nothing here is a guard. A segment is written when a run starts or is extended,
and read when something needs to know what was in force; no comparison is made
against it and nothing refuses to run because of it. The one thing it must not
do is lose history, which is why appending is the only write.

Each segment also records the commit that produced it. A continuation after a
week of edits runs different code against the same run, and the segment is the
only place that is visible.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
CONFIG_FILENAME = "config.json"


class RunConfigError(Exception):
    """The stored configuration history cannot be read or extended."""


@dataclass(frozen=True)
class ConfigSegment:
    """One stretch of a run's history, from ``from_step`` until the next segment."""

    from_step: int
    profile: str
    config: dict[str, Any]
    overrides: dict[str, str] = field(default_factory=dict)
    git_commit: str | None = None
    git_dirty: bool | None = None
    recorded_at: str | None = None

    def document(self) -> dict[str, Any]:
        return {
            "from_step": self.from_step,
            "profile": self.profile,
            "overrides": dict(self.overrides),
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "recorded_at": self.recorded_at,
            "config": self.config,
        }

    @classmethod
    def from_document(cls, payload: dict[str, Any]) -> ConfigSegment:
        return cls(
            from_step=int(payload["from_step"]),
            profile=str(payload["profile"]),
            config=payload.get("config") or {},
            overrides=dict(payload.get("overrides") or {}),
            git_commit=payload.get("git_commit"),
            git_dirty=payload.get("git_dirty"),
            recorded_at=payload.get("recorded_at"),
        )


def config_path(run_dir: Path | str) -> Path:
    return Path(run_dir) / CONFIG_FILENAME


def read_segments(run_dir: Path | str) -> tuple[ConfigSegment, ...]:
    """Every segment in order, or empty for a run that predates this file."""

    path = config_path(run_dir)
    if not path.is_file():
        return ()
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise RunConfigError(f"{path} is not readable JSON: {error}") from error
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise RunConfigError(f"{path} has schema_version {version!r}, expected {SCHEMA_VERSION}")
    segments = tuple(ConfigSegment.from_document(row) for row in payload.get("segments", ()))
    steps = [segment.from_step for segment in segments]
    if steps != sorted(steps):
        raise RunConfigError(f"{path} segments are not in step order")
    return segments


def append_segment(run_dir: Path | str, segment: ConfigSegment) -> Path:
    """Add one segment, refusing to rewrite or reorder what is already there.

    A repeat of the newest step replaces it: relaunching a run that has not yet
    trained past ``from_step`` is a correction, not a second era. Anything
    earlier is rejected, because it would make the history disagree with the
    checkpoints beside it.
    """

    path = config_path(run_dir)
    existing = list(read_segments(run_dir))
    if existing:
        newest = existing[-1].from_step
        if segment.from_step < newest:
            raise RunConfigError(
                f"cannot record a segment at step {segment.from_step}: "
                f"{path} already has one at {newest}"
            )
        if segment.from_step == newest:
            existing.pop()
    existing.append(segment)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "segments": [item.document() for item in existing],
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    temporary.replace(path)
    return path


def config_at(run_dir: Path | str, step: int) -> ConfigSegment | None:
    """The segment in force at ``step``, or ``None`` if nothing is recorded."""

    segments = read_segments(run_dir)
    if not segments:
        return None
    index = bisect_right([segment.from_step for segment in segments], step)
    # A step before the first segment reads as the first: a run cannot have
    # trained under a configuration it had not been given yet.
    return segments[max(index - 1, 0)]


def latest_config(run_dir: Path | str) -> ConfigSegment | None:
    """The newest segment -- what a final checkpoint was produced under."""

    segments = read_segments(run_dir)
    return segments[-1] if segments else None
