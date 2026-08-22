"""The figure set is what a run can show; the manifest is what the docs show.

Two questions, deliberately answered in two places. These tests hold the two in
agreement so the split does not become drift: every chart the manifest publishes
has to be one the figure set knows how to produce, and publication has to get it
by copying that run's rendered copy rather than by rendering its own.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from boost_and_broadside.publication import renderers  # noqa: F401  (registers them)
from boost_and_broadside.publication.figure_set import (
    FIGURES,
    FIGURES_BY_NAME,
    required_artifact_types,
)
from boost_and_broadside.publication.manifest import load_manifest
from boost_and_broadside.publication.renderer_api import PublicationError, get_renderer

_ROOT = Path(__file__).resolve().parents[2]


def _computed_entries():
    """Manifest entries rendered from artifacts, i.e. everything but media and assets."""

    return [entry for entry in load_manifest(_ROOT).entries if entry.artifacts]


def test_every_figure_names_a_registered_renderer_and_its_required_sources():
    for figure in FIGURES:
        renderer = get_renderer(figure.renderer)
        assert set(renderer.required_artifacts) <= set(figure.sources), figure.name
        assert set(figure.sources) <= set(renderer.source_names), figure.name


def test_figure_names_are_unique():
    assert len(FIGURES_BY_NAME) == len(FIGURES)


def test_the_set_covers_every_computed_publication():
    """A published figure with no entry here could not be reproduced for a new run."""

    missing = [
        entry.name
        for entry in _computed_entries()
        if entry.figure is not None and entry.figure not in FIGURES_BY_NAME
    ]
    assert missing == []


def test_no_publication_renders_a_chart_the_figure_set_already_renders():
    """Publishing copies; it does not render a second time.

    Two independent renders of one chart agree only by convention, and nothing
    compares them, so a manifest entry naming a chart renderer directly could
    put one image in a run's own evidence and another in `docs/` under the same
    claim. The published output is a copy of the run's, or it is not published.
    """

    chart_renderers = {figure.renderer for figure in FIGURES}
    rendered_twice = [
        entry.name for entry in _computed_entries() if entry.renderer_name in chart_renderers
    ]
    assert rendered_twice == []


def test_every_published_figure_is_copied_from_one_runs_figure_set():
    for entry in _computed_entries():
        assert set(entry.artifacts) == {"figures"}, entry.name
        assert entry.figure is not None, entry.name
        assert entry.artifacts["figures"].endswith("/artifacts/figures"), entry.name


def test_required_artifact_types_are_what_a_finished_run_produces():
    """The evaluation campaign for a new run is exactly this list."""

    assert set(required_artifact_types()) == {
        "elo-calibration",
        "wandb-export",
        "crossover",
        "elo-scale",
        "semi-random-ladder",
        "ar-report",
        "noise-calibration",
    }


def test_rendering_a_run_without_its_measurements_says_which_one_is_missing(tmp_path):
    from boost_and_broadside.modes.figures import render_run_figures

    (tmp_path / "empty-run").mkdir()

    with pytest.raises(PublicationError, match="no crossover artifact"):
        render_run_figures("empty-run", checkpoint_dir=str(tmp_path), only=("crossover_ratio.png",))


def test_selecting_an_unknown_figure_is_refused(tmp_path):
    from boost_and_broadside.modes.figures import render_run_figures

    (tmp_path / "empty-run").mkdir()

    with pytest.raises(PublicationError, match="no figure matches"):
        render_run_figures("empty-run", checkpoint_dir=str(tmp_path), only=("nope.png",))
