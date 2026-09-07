"""The figure set is the complete list of charts a finished run can produce.

It is the only such list. The documents link at a run's rendered figures rather
than holding copies, so there is nothing to hold in agreement with it -- what
these tests protect is that every entry is renderable and that the campaign of
measurements a new run needs is stated in one place.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from boost_and_broadside.charts import renderers  # noqa: F401  (registers them)
from boost_and_broadside.charts.figure_set import (
    FIGURES,
    FIGURES_BY_NAME,
    required_artifact_types,
)
from boost_and_broadside.charts.renderer_api import PublicationError, get_renderer

_ROOT = Path(__file__).resolve().parents[2]

# `](.../artifacts/figures/<name>)` in any tracked document.
_FIGURE_LINK = re.compile(r"artifacts/figures/([A-Za-z0-9_.]+/?)\)")


def test_every_figure_names_a_registered_renderer_and_its_required_sources():
    for figure in FIGURES:
        renderer = get_renderer(figure.renderer)
        assert set(renderer.required_artifacts) <= set(figure.sources), figure.name
        assert set(figure.sources) <= set(renderer.source_names), figure.name


def test_figure_names_are_unique():
    assert len(FIGURES_BY_NAME) == len(FIGURES)


def test_the_documents_link_at_figures_the_set_actually_renders():
    """A `docs/` link into the reference run must name a chart that exists.

    The pages link at `checkpoints/<run>/artifacts/figures/<name>` rather than
    holding copies, so a renamed figure breaks the documents silently -- the
    link is just a path, and nothing else would notice.
    """

    linked = set()
    for page in (_ROOT / "docs").rglob("*.md"):
        linked.update(_FIGURE_LINK.findall(page.read_text()))
    linked.update(_FIGURE_LINK.findall((_ROOT / "README.md").read_text()))

    assert linked, "no document links at the reference run's figures"
    renderable = set(FIGURES_BY_NAME) | {
        f"{figure.name}/" for figure in FIGURES if get_renderer(figure.renderer).multi_file
    }
    # The artifact's own manifest is what a reader follows to see which
    # measurement each chart came from, so linking it is expected.
    renderable.add("artifact.json")
    assert linked <= renderable, sorted(linked - renderable)


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
