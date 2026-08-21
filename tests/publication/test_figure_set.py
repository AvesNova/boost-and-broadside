"""The figure set is what a run can show; the manifest is what the docs show.

Two questions, deliberately answered in two places. These tests hold the two in
agreement so the split does not become drift: every computed figure the manifest
publishes has to exist in the set, rendered the same way from the same kinds of
measurement.
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

    by_renderer = {figure.renderer: figure for figure in FIGURES}
    missing = [
        entry.name for entry in _computed_entries() if entry.renderer_name not in by_renderer
    ]
    assert missing == []


def test_published_figures_read_the_same_artifact_kinds_the_set_declares():
    """The manifest pins exact artifact ids; the set pins their types. The source
    *names* must still line up, or a repointed manifest would feed a renderer
    something the set never intended."""

    by_renderer = {figure.renderer: figure for figure in FIGURES}
    for entry in _computed_entries():
        figure = by_renderer[entry.renderer_name]
        assert set(entry.artifacts) == set(figure.sources), entry.name


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
