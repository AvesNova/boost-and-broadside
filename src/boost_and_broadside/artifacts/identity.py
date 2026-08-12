"""Artifact recipes and the identities derived from them.

A *recipe* is the complete description of what a measurement was asked to do:
its exact subjects, its parameters, and the artifacts it reads. Two runs with
the same recipe measure the same thing; two runs with different recipes do not,
and must never land in the same directory.

The identity that follows from a recipe combines a readable creation time with a
short digest of the recipe itself. The time keeps repeated measurements of one
recipe distinct — each has its own seeds, machine, and provenance — while the
digest makes it obvious at a glance which directories describe the same request.
Resume matches on the full digest and then re-verifies the recipe field by field,
so a truncated digest can never silently continue the wrong measurement.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from boost_and_broadside.config.fingerprint import canonical_data, fingerprint

ARTIFACT_MANIFEST_SCHEMA_VERSION = 1

_TYPE_PATTERN = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")
_SHORT_DIGEST_LENGTH = 8


@dataclass(frozen=True)
class ArtifactRecipe:
    """What a measurement was asked to do, in a form that hashes stably."""

    artifact_type: str
    result_schema_version: int
    subjects: Mapping[str, Any] = field(default_factory=dict)
    parameters: Mapping[str, Any] = field(default_factory=dict)
    sources: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _TYPE_PATTERN.match(self.artifact_type):
            raise ValueError(
                f"artifact type {self.artifact_type!r} must be lowercase and hyphenated"
            )
        if self.result_schema_version < 1:
            raise ValueError("result_schema_version must be a positive integer")

    def canonical(self) -> dict[str, Any]:
        """Deterministic JSON data for hashing and for the stored manifest."""

        return {
            "artifact_type": self.artifact_type,
            "result_schema_version": self.result_schema_version,
            "subjects": canonical_data(self.subjects),
            "parameters": canonical_data(self.parameters),
            "sources": canonical_data(self.sources),
        }

    def digest(self) -> str:
        """The full SHA-256 of the canonical recipe."""

        return fingerprint(self.canonical())

    def short_digest(self) -> str:
        """The readable prefix used in artifact directory names."""

        return self.digest()[:_SHORT_DIGEST_LENGTH]


def artifact_id(recipe: ArtifactRecipe, created_at: datetime) -> str:
    """``20260809T142500Z-a81bc39e`` — creation instant plus recipe digest."""

    stamp = created_at.strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{recipe.short_digest()}"


def recipe_from_document(document: Mapping[str, Any]) -> ArtifactRecipe:
    """Rebuild a recipe from a stored manifest for verification and reporting."""

    return ArtifactRecipe(
        artifact_type=document["artifact_type"],
        result_schema_version=document["result_schema_version"],
        subjects=document.get("subjects", {}),
        parameters=document.get("parameters", {}),
        sources=document.get("sources", {}),
    )


__all__ = [
    "ARTIFACT_MANIFEST_SCHEMA_VERSION",
    "ArtifactRecipe",
    "artifact_id",
    "recipe_from_document",
]
