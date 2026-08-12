"""The output taxonomy, enforced rather than documented.

Compute modes write artifacts; publication writes ``docs/``; capture writes
scratch. The rule that matters most is the one a future change is most likely to
break by accident — a mode quietly rendering into ``docs/`` again — so it is
checked against the source, not against a run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_SOURCE = Path(__file__).resolve().parents[2] / "src" / "boost_and_broadside"
_MODES = sorted((_SOURCE / "modes").glob("*.py"))


@pytest.mark.parametrize("module", _MODES, ids=lambda path: path.stem)
def test_no_compute_mode_names_the_docs_tree(module: Path) -> None:
    source = module.read_text()
    offending = [
        line.strip()
        for line in source.splitlines()
        if '"docs/' in line or "'docs/" in line or "Path(\"docs" in line
    ]
    assert not offending, f"{module.name} still targets docs/: {offending}"


@pytest.mark.parametrize("module", _MODES, ids=lambda path: path.stem)
def test_no_compute_mode_renders_figures(module: Path) -> None:
    source = module.read_text()
    assert "matplotlib" not in source, f"{module.name} renders figures; publication owns rendering"


def test_every_artifact_writing_mode_accepts_an_injected_store() -> None:
    """A store parameter is what lets smoke and tests redirect managed roots."""

    expected = {
        "ar_report": "run_ar_report_mode",
        "crossover": "run_crossover_mode",
        "elo_calibrate": "run_elo_calibrate_mode",
        "elo_scale": "run_elo_scale_mode",
        "feature_stats": "run_feature_stats_mode",
        "noise_calibration": "run_noise_calibration_mode",
        "semi_random_tournament": "run_semi_random_tournament",
    }
    import importlib
    import inspect

    for module_name, function_name in expected.items():
        module = importlib.import_module(f"boost_and_broadside.modes.{module_name}")
        signature = inspect.signature(getattr(module, function_name))
        assert "store" in signature.parameters, f"{module_name}.{function_name} has no store"
