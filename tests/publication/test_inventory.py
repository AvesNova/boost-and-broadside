"""The shipped inventory: nothing published is missing from it, and it parses.

The manifest is the only place that decides what belongs under ``docs/``. This
holds it to that claim against the S01 record of what is actually tracked, so a
canonical output cannot quietly stop being owned by anyone.
"""

from __future__ import annotations

import json
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


def test_the_repository_checks_clean_with_nothing_selected_yet() -> None:
    report = run_publish(_ROOT, check=True)

    assert not report.failed
    assert report.by_status(UNSELECTED)
    assert not (_ROOT / "docs" / "results" / "provenance.md").exists()
