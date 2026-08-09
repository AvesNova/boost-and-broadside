"""S01 characterization seams for the mode-system refactor.

These lock the old dispatcher and profile behavior only until the plan explicitly
replaces it.  The serialized files are evidence, not an endorsement of legacy
defaults (in particular, ``bc-stale`` is intentionally stale evidence for S11).
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import os
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Any

import pytest

import main
from boost_and_broadside.modes.capture import _find_run_dir, parse_matchup
from boost_and_broadside.modes.elo_stats import find_run_dir
from runs.bc import BC_TRAIN_CONFIG
from runs.rl import RL_TRAIN_CONFIG
from runs.rl_fields import RL_FIELDS_TRAIN_CONFIG
from runs.shared import MODEL_CONFIG, SHIP_CONFIG

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
        ("bc-stale", BC_TRAIN_CONFIG),
    ],
)
def test_profile_snapshot_matches_pre_refactor_baseline(name, config):
    """Keep RL/RL-fields mechanically identical through S02; preserve BC evidence."""
    expected = json.loads((_SNAPSHOTS / f"{name}.json").read_text())
    assert _profile_snapshot(name, config) == expected


def _capture_parser(monkeypatch) -> argparse.ArgumentParser:
    captured: list[argparse.ArgumentParser] = []
    original = argparse.ArgumentParser.parse_args

    def parse_args(parser, *args, **kwargs):
        captured.append(parser)
        return original(parser, *args, **kwargs)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_args)
    monkeypatch.setattr(sys, "argv", ["main.py"])
    main._parse_args()
    assert len(captured) == 1
    return captured[0]


def _parser_actions(parser: argparse.ArgumentParser) -> list[dict[str, Any]]:
    actions = []
    for action in parser._actions:
        if not action.option_strings:
            continue
        actions.append(
            {
                "options": action.option_strings,
                "dest": action.dest,
                "default": str(action.default)
                if isinstance(action.default, Path)
                else action.default,
                "const": str(action.const) if isinstance(action.const, Path) else action.const,
                "nargs": action.nargs,
                "required": action.required,
                "choices": sorted(action.choices) if action.choices is not None else None,
            }
        )
    return actions


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


def test_machine_readable_inventory_matches_current_parser_and_assets(monkeypatch):
    inventory = json.loads(_INVENTORY.read_text())
    assert inventory["parser_actions"] == _parser_actions(_capture_parser(monkeypatch))
    assert inventory["published_assets"] == _tracked_publication_assets()


def test_legacy_resume_selection_uses_optional_path_and_newest_mtime(tmp_path):
    older = tmp_path / "older" / "step_000000200.pt"
    newer = tmp_path / "newer" / "step_000000100.pt"
    older.parent.mkdir()
    newer.parent.mkdir()
    older.touch()
    newer.touch()
    os.utime(older, (100, 100))
    os.utime(newer, (200, 200))
    (newer.parent / "wandb_run_id.txt").write_text("wandb-new\n")

    assert main._find_resume_checkpoint("", str(tmp_path)) == (str(newer), "wandb-new")
    assert main._find_resume_checkpoint(str(older.parent)) == (str(older), None)
    assert main._find_resume_checkpoint(str(older)) == (str(older), None)


def test_legacy_run_selection_treats_capture_none_as_latest_and_elo_latest_includes_empty_dirs(
    tmp_path,
):
    no_checkpoint = tmp_path / "empty"
    old = tmp_path / "old"
    newest = tmp_path / "newest"
    for run in (no_checkpoint, old, newest):
        run.mkdir()
    old_checkpoint = old / "step_000000200.pt"
    new_checkpoint = newest / "step_000000100.pt"
    old_checkpoint.touch()
    new_checkpoint.touch()
    os.utime(old_checkpoint, (100, 100))
    os.utime(new_checkpoint, (200, 200))

    assert _find_run_dir("none", str(tmp_path)) == newest
    assert _find_run_dir("latest", str(tmp_path)) == newest
    assert find_run_dir("latest", str(tmp_path)) == newest
    assert _find_run_dir("old", str(tmp_path)) == old
    assert find_run_dir("empty", str(tmp_path)) == no_checkpoint


@pytest.mark.parametrize(
    ("spec", "expected"),
    [("4", (4, 4)), ("4v4", (4, 4)), ("3v5", (3, 5))],
)
def test_capture_size_parser_preserves_legacy_symmetric_and_asymmetric_forms(spec, expected):
    assert parse_matchup(spec) == expected


def test_main_dispatch_retains_legacy_output_defaults(monkeypatch):
    ar_calls: list[dict[str, Any]] = []
    captured: dict[str, Any] = {}

    monkeypatch.setattr(main, "run_ar_report_mode", lambda **kwargs: ar_calls.append(kwargs))
    monkeypatch.setattr(sys, "argv", ["main.py", "--mode", "ar_report"])
    main.main()
    assert [call["out_dir"] for call in ar_calls] == ["docs/ar_report/2v2", "docs/ar_report/1v1"]

    monkeypatch.setattr(main, "run_capture_mode", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(sys, "argv", ["main.py", "--mode", "capture"])
    main.main()
    assert captured["run_spec"] == "resilient-resonance-682"
    assert captured["out_dir"] == Path("gameplay_clips")
