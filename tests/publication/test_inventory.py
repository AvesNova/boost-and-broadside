"""The shipped inventory: nothing published is missing from it, and it parses.

The manifest is the only place that decides what belongs under ``docs/``. This
holds it to that claim against the tree as it stands, so a result file cannot be
added without an owner and cannot quietly stop having one.

``bnb publish --check`` covers the other direction: a declared output that goes
missing or changes fails there. What it cannot see is a file nobody declared, so
that is what the ownership test below is for.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from boost_and_broadside.publication import registered_renderers
from boost_and_broadside.publication.manifest import load_manifest
from boost_and_broadside.publication.publish import UNSELECTED, run_publish

_ROOT = Path(__file__).resolve().parents[2]

# Tracked locations under docs/ that hold results rather than prose. A file here
# is a published artifact and needs a manifest entry that owns it.
_RESULT_PATHS = (
    "docs/results/",
    "docs/ar_report/",
    "docs/noise_calibration/",
    "docs/crossover/",
)
_RESULT_FILES = ("docs/policy_architecture.png",)

# `bnb publish` generates these two itself, as the index and ownership record of
# everything it published. They are outputs of the manifest as a whole rather
# than of any one entry, so no entry declares them and none should.
_GENERATED_INDEX = ("docs/results/provenance.md", "docs/results/provenance.json")


def _tracked_result_assets() -> list[str]:
    """Every result file Git currently tracks under docs/."""

    tracked = subprocess.run(
        ["git", "ls-files", "docs"],
        check=True,
        capture_output=True,
        text=True,
        cwd=_ROOT,
    ).stdout.split()
    return sorted(
        path
        for path in tracked
        if (path in _RESULT_FILES or path.startswith(_RESULT_PATHS))
        and path not in _GENERATED_INDEX
    )


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(_ROOT)


def test_the_shipped_manifest_parses_and_names_registered_renderers(manifest) -> None:
    known = set(registered_renderers())
    assert manifest.entries
    assert {entry.renderer_name for entry in manifest.entries} <= known


def test_every_declared_output_is_unique_and_inside_docs(manifest) -> None:
    outputs = [entry.output for entry in manifest.entries]
    assert len(outputs) == len(set(outputs))
    assert all(output.startswith("docs/") for output in outputs)


def test_every_tracked_result_asset_is_owned_by_a_manifest_entry(manifest) -> None:
    """A result file in the tree that no entry declares is unowned and unpublished.

    Scanning the tree rather than a frozen list is what makes this catch a figure
    added today and never declared. The generated provenance index is the record
    of what publication owns, so an unowned file is invisible to it and would be
    pruned if it were ever adopted and then dropped.
    """

    owners = {entry.output for entry in manifest.entries}

    unowned = [
        path
        for path in _tracked_result_assets()
        if path not in owners and not any(path.startswith(f"{owner}/") for owner in owners)
    ]
    assert unowned == []


def test_every_entry_has_a_source(manifest) -> None:
    """S16 selected the reference run's artifacts; nothing is pending a source.

    An entry without one is not an error in general — it is how the inventory
    carries an output whose measurement has not been made yet — but the shipped
    manifest has no such entry any more, and a new one would mean a canonical
    output silently stopped being regenerated.
    """

    unselected = [entry.name for entry in manifest.entries if not entry.selected]
    assert unselected == []


def test_every_selected_artifacts_payload_is_tracked_or_a_raw_sample(manifest) -> None:
    """A clean clone must hold every byte the selected artifacts are verified against.

    ``verify_artifact`` re-hashes each file ``artifact.json`` records and exempts
    exactly one thing: a pruned ``samples/`` payload, which is local by design.
    Everything else it names has to be in the index, or ``publish --check``
    passes here and fails on a fresh clone — which is precisely what happened
    when ``.gitignore``'s repository-wide ``*.log`` reached into the landmark
    W&B export and took ``files/output.log`` out of the index.
    """

    tracked = set(
        subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split("\0")
    )

    untracked = []
    for entry in manifest.entries:
        if not entry.selected:
            continue
        for location in entry.artifacts.values():
            recorded = json.loads((_ROOT / location / "artifact.json").read_text())["files"]
            untracked.extend(
                f"{location}/{record['path']}"
                for record in recorded
                if not record["path"].startswith("samples/")
                and f"{location}/{record['path']}" not in tracked
            )
    assert sorted(set(untracked)) == []


@pytest.mark.slow
def test_the_repository_checks_clean_against_its_selected_sources() -> None:
    """Deselected by default: this re-renders every published figure.

    It was the second-slowest test in the suite at 45s, paid on every commit to
    verify a tree that changes a few times a year. ``pytest -m slow`` before
    publishing is the same check at the moment it can actually fail.
    """

    report = run_publish(_ROOT, check=True)

    assert not report.failed
    assert not report.by_status(UNSELECTED)
    assert (_ROOT / "docs" / "results" / "provenance.md").exists()
