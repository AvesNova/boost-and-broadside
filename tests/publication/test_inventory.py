"""The shipped inventory: nothing published is missing from it, and it parses.

The manifest is the only place that decides what belongs under ``docs/``. This
holds it to that claim against the S01 record of what is actually tracked, so a
canonical output cannot quietly stop being owned by anyone.
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
_INVENTORY = _ROOT / "docs" / "internal" / "mode-characterization.json"

# S08 replaced the 2v2/1v1 AR report pair with one canonical 4v4 report. Their
# files are still tracked and are deleted by the documentation sweep in S18; the
# manifest deliberately does not adopt outputs that are on their way out.
_RETIRED = ("docs/ar_report/1v1/", "docs/ar_report/2v2/")


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


def test_every_tracked_published_asset_is_owned_by_exactly_one_entry(manifest) -> None:
    tracked = [
        path
        for path in json.loads(_INVENTORY.read_text())["published_assets"]
        if not path.startswith(_RETIRED)
    ]
    owners = {entry.output for entry in manifest.entries}

    unowned = [
        path
        for path in tracked
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


def test_the_repository_checks_clean_against_its_selected_sources() -> None:
    report = run_publish(_ROOT, check=True)

    assert not report.failed
    assert not report.by_status(UNSELECTED)
    assert (_ROOT / "docs" / "results" / "provenance.md").exists()
