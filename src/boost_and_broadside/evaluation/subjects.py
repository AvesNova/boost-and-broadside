"""Exact descriptions of the agents a measurement was actually about.

An artifact that records ``--team0 checkpoints/x/step_900.pt`` has recorded a
path, not a subject: the same path can hold different weights next week. What
makes a measurement citable is the file's content hash, the step it claims, and
— for the scripted controller, which is a *configured* opponent rather than a
fixed one — the configuration it played under.

These descriptions go into artifact recipes, so they must be stable, cheap, and
free of absolute machine paths.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.artifacts.store import file_sha256

_POLICY_SUFFIX = ".pt"


def describe_agent(
    spec: str,
    *,
    global_step: int | None = None,
    checkpoint_root: str | Path = "checkpoints",
) -> dict[str, Any]:
    """Describe one resolved agent spec well enough to identify it later."""

    if not spec.endswith(_POLICY_SUFFIX):
        described: dict[str, Any] = {"kind": spec}
        if spec in {"scripted", "scripted_team"}:
            described["scripted_config"] = dataclasses.asdict(StochasticAgentConfig())
        return described

    path = Path(spec)
    root = Path(checkpoint_root).resolve()
    resolved = path.resolve()
    return {
        "kind": "policy",
        "checkpoint": (
            str(resolved.relative_to(root)) if resolved.is_relative_to(root) else str(path)
        ),
        "sha256": file_sha256(path),
        "global_step": global_step,
    }


def describe_checkpoint_configuration(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The training configuration a checkpoint was produced under, by name.

    A checkpoint carries its complete resolved configuration and the run keeps a
    ``config.json`` history beside it; repeating either in every artifact would
    be noise. Two measurements taken against the same run and profile are
    comparable, and anything finer is a question for those two records, which
    state values rather than digests of values.
    """

    recorded = payload.get("resolved_config") or {}
    return {"profile": recorded.get("profile")}


def describe_agents(
    *, checkpoint_root: str | Path = "checkpoints", **specs: str
) -> dict[str, Any]:
    """Describe several named agent specs, e.g. ``team0=..., team1=...``."""

    return {
        name: describe_agent(spec, checkpoint_root=checkpoint_root)
        for name, spec in specs.items()
    }


def describe_environment(env_config: Any, *, matchups: Any = None) -> dict[str, Any]:
    """The environment shape a measurement ran under, as recipe parameters."""

    described: dict[str, Any] = {
        "num_ships": env_config.num_ships,
        "num_fields": env_config.num_fields,
        "max_bullets": env_config.max_bullets,
        "max_episode_steps": env_config.max_episode_steps,
    }
    if matchups is not None:
        described["matchups"] = [f"{team0}v{team1}" for team0, team1 in matchups]
    return described


def describe_source_artifact(manifest: Mapping[str, Any], digest: str) -> dict[str, Any]:
    """Identify an artifact a reanalysis was derived from."""

    return {
        "artifact_id": manifest["artifact_id"],
        "artifact_type": manifest["artifact_type"],
        "recipe_digest": manifest["recipe_digest"],
        "digest": digest,
    }


__all__ = [
    "describe_agent",
    "describe_agents",
    "describe_environment",
    "describe_source_artifact",
]
