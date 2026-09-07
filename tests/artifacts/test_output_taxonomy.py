"""The seam that lets a mode's artifact root be redirected.

Two source-grep tests used to live here, asserting that no mode module contained
the string ``"docs/`` or the string ``matplotlib``. They encoded an architecture
rule as text matching, which fires on a rename and misses any spelling nobody
thought of. The taxonomy they described is real; the check was theatre.
"""

from __future__ import annotations

from pathlib import Path

_SOURCE = Path(__file__).resolve().parents[2] / "src" / "boost_and_broadside"


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
