"""S01 characterization seams for the mode-system refactor.

These lock the old dispatcher and profile behavior only until the plan explicitly
replaces it.  The serialized files are evidence, not an endorsement of legacy
defaults: ``bc-stale.json`` records the pre-correction BC configuration, which
S11 has since deliberately replaced.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any

import pytest

from boost_and_broadside.config.defaults import MODEL_CONFIG, SHIP_CONFIG
from boost_and_broadside.evaluation.sizes import parse_matchup
from boost_and_broadside.profiles import (
    RL_FIELDS_TRAIN_CONFIG,
    RL_TRAIN_CONFIG,
)

_ROOT = Path(__file__).resolve().parents[1]
_SNAPSHOTS = _ROOT / "tests" / "fixtures" / "mode_refactor"
_INVENTORY = _ROOT / "docs" / "internal" / "mode-characterization.json"


def _normalize(value: Any) -> Any:
    """Turn current dataclass config into stable, complete JSON evidence.

    S01 has no resolver yet.  Schedule closures are therefore represented by
    their defining callable and captured construction values, rather than by a
    partial sample of values at arbitrary training steps.
    """
    if dataclasses.is_dataclass(value):
        return {
            field.name: _normalize(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if inspect.isfunction(value):
        return {
            "callable": f"{value.__module__}.{value.__qualname__}",
            "closure": {
                name: _normalize(cell.cell_contents)
                for name, cell in zip(
                    value.__code__.co_freevars, value.__closure__ or (), strict=True
                )
            },
        }
    if isinstance(value, Enum):
        return {
            "enum": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": value.value,
        }
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_normalize(item) for item in value]
    if isinstance(value, (frozenset, set)):
        return [_normalize(item) for item in sorted(value)]
    return value


def _profile_snapshot(name: str, config: Any) -> dict[str, Any]:
    return {
        "profile": name,
        "ship_config": _normalize(SHIP_CONFIG),
        "model_config": _normalize(MODEL_CONFIG),
        "train_config": _normalize(config),
    }


@pytest.mark.parametrize(
    ("name", "config"),
    [
        ("rl", RL_TRAIN_CONFIG),
        ("rl-fields", RL_FIELDS_TRAIN_CONFIG),
    ],
)
def test_profile_snapshot_matches_pre_refactor_baseline(name, config):
    """Keep RL and RL-fields mechanically identical to their pre-refactor form.

    BC left this parametrization in S11, which corrected it deliberately.
    ``bc-stale.json`` stays as the "before" side of that review and is asserted
    by ``tests/config/test_bc_profile.py``.
    """
    expected = json.loads((_SNAPSHOTS / f"{name}.json").read_text())
    assert _profile_snapshot(name, config) == expected


def _tracked_publication_assets() -> list[str]:
    tracked = subprocess.run(
        ["git", "ls-files", "docs"], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return sorted(
        path
        for path in tracked
        if path == "docs/policy_architecture.png"
        or path == "docs/crossover/crossover.json"
        or path.startswith("docs/ar_report/")
        or path.startswith("docs/noise_calibration/")
        or path.startswith("docs/results/")
    )


def test_machine_readable_inventory_remains_as_legacy_evidence_and_tracks_assets():
    """No published asset recorded at the branch base has been lost since.

    This was an equality until S16, which selected the reference run's artifacts
    and regenerated the inventory: publishing added the canonical 4v4 AR report
    and the generated provenance index. Growth is allowed — a canonical output
    *disappearing* is what this guards, and the S01 record stays frozen as the
    thing it is compared against.
    """

    inventory = json.loads(_INVENTORY.read_text())
    assert any(action["dest"] == "mode" for action in inventory["parser_actions"])
    assert set(inventory["published_assets"]) <= set(_tracked_publication_assets())


@pytest.mark.parametrize(
    ("spec", "expected"),
    [("4", (4, 4)), ("4v4", (4, 4)), ("3v5", (3, 5))],
)
def test_capture_size_parser_preserves_legacy_symmetric_and_asymmetric_forms(spec, expected):
    assert parse_matchup(spec) == expected
