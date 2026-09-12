"""The house style's placement helpers, tested where the figures cannot see them.

``viz/style.py`` is the one definition of what every figure in the project looks
like, and it had no tests of its own. What it does went unchecked because the
only way to reach it was to render a chart and look at the chart, and a chart
that is wrong off the edge of its own canvas looks exactly like a chart that is
right.

That is not hypothetical. ``label_series_ends`` maps data coordinates into
axes-fraction space to space colliding labels by how far apart they *look*.
Matplotlib autoscales lazily, so on a freshly plotted axes that transform is
still the default identity and a rating of 1000 was read as axes-fraction 1000 --
a label a thousand panel-heights above the panel, which ``bbox_inches="tight"``
then faithfully included. ``tie_conventions.png`` shipped at 1.4 gigapixels and
nothing raised.

So these assert positions, in axes fraction, rather than rendering anything.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from boost_and_broadside.viz import style  # noqa: E402

# The de-collision gap, in fraction of the panel. Named here because both the
# spacing test and the in-panel bound are stated against it.
_MINIMUM_GAP = 0.034


@pytest.fixture
def axes():
    figure = style.new_figure((6.0, 4.0))
    yield figure.add_subplot(111)
    plt.close(figure)


def _label_fractions(axes) -> list[float]:
    """Each annotation's y, in axes fraction -- the units it was placed in."""

    return [annotation.xy[1] for annotation in axes.texts]


def _elo_series(axes) -> list[tuple[float, float, str, str]]:
    """Two converging curves on a rating scale, as a calibration figure has."""

    axes.plot([0.0, 1.0, 2.0], [200.0, 600.0, 1000.0])
    axes.plot([0.0, 1.0, 2.0], [210.0, 605.0, 1002.0])
    return [
        (2.0, 1000.0, style.TRAINING, "live"),
        (2.0, 1002.0, style.CALIBRATED, "calibrated"),
    ]


def test_labels_land_inside_the_panel_on_an_unsettled_axes(axes) -> None:
    """The bug, stated directly: no caller has to draw the figure first.

    Before the fix every one of these came back at roughly the data value --
    axes-fraction 1000 -- because the lazily-autoscaled transform was still the
    identity. A fraction is a fraction; anything far outside [0, 1] is off the
    page.
    """

    style.label_series_ends(axes, _elo_series(axes))

    fractions = _label_fractions(axes)
    assert len(fractions) == 2
    for fraction in fractions:
        assert -_MINIMUM_GAP <= fraction <= 1.0 + _MINIMUM_GAP, fractions


def test_settling_the_axes_first_changes_nothing(axes) -> None:
    """Two callers already drew the canvas before labelling. They must still be
    right, and they must agree with the callers that do not."""

    entries = _elo_series(axes)
    axes.figure.canvas.draw()
    style.label_series_ends(axes, entries)
    settled = _label_fractions(axes)

    second = style.new_figure((6.0, 4.0))
    try:
        fresh_axes = second.add_subplot(111)
        style.label_series_ends(fresh_axes, _elo_series(fresh_axes))
        unsettled = _label_fractions(fresh_axes)
    finally:
        plt.close(second)

    assert settled == pytest.approx(unsettled, abs=1e-9)


def test_converging_labels_are_nudged_apart_not_stacked(axes) -> None:
    """The helper's whole reason to exist: two curves that end together would
    otherwise print one label on top of the other."""

    axes.plot([0.0, 1.0], [100.0, 500.0])
    style.label_series_ends(
        axes,
        [
            (1.0, 500.0, style.TRAINING, "live"),
            (1.0, 500.0, style.CALIBRATED, "calibrated"),
            (1.0, 500.0, style.AVG, "avg"),
        ],
    )

    fractions = sorted(_label_fractions(axes))
    assert len(fractions) == 3
    gaps = [b - a for a, b in zip(fractions, fractions[1:])]
    assert all(gap >= _MINIMUM_GAP - 1e-9 for gap in gaps), fractions


def test_a_label_that_needs_no_room_is_not_moved(axes) -> None:
    """Nudging is for collisions. Series that end far apart keep their own y, or
    the label stops pointing at the curve it names."""

    axes.plot([0.0, 1.0], [0.0, 100.0])
    axes.set_ylim(0.0, 100.0)
    style.label_series_ends(
        axes,
        [(1.0, 10.0, style.TRAINING, "low"), (1.0, 90.0, style.CALIBRATED, "high")],
    )

    assert sorted(_label_fractions(axes)) == pytest.approx([0.1, 0.9], abs=1e-6)


def test_the_gap_is_measured_in_what_the_reader_sees_not_in_the_data(axes) -> None:
    """The docstring's claim about a log axis. Equal pixel gaps span wildly
    different data distances there, so de-collision has to happen after the
    scale, not before it."""

    axes.set_yscale("log")
    axes.plot([0.0, 1.0], [1.0, 10_000.0])
    style.label_series_ends(
        axes,
        [
            (1.0, 9_000.0, style.TRAINING, "near"),
            (1.0, 10_000.0, style.CALIBRATED, "nearer"),
        ],
    )

    fractions = sorted(_label_fractions(axes))
    # A decade apart in data is a wide gap in pixels, so neither is nudged;
    # 9000 and 10000 are close on a log scale and both stay in the panel.
    assert all(-_MINIMUM_GAP <= fraction <= 1.0 + _MINIMUM_GAP for fraction in fractions)
    assert fractions[1] - fractions[0] >= _MINIMUM_GAP - 1e-9


def test_no_entries_draws_nothing(axes) -> None:
    style.label_series_ends(axes, [])

    assert len(axes.texts) == 0


def test_a_reference_line_is_labelled_at_its_own_value(axes) -> None:
    """The landmark rule and its label have to agree, or the figure says one
    number and shows another."""

    axes.plot([0.0, 1.0], [500.0, 1500.0])
    style.draw_reference_lines(axes, [("scripted controller", 1000.0)])

    line = axes.lines[-1]
    assert line.get_ydata() == pytest.approx([1000.0, 1000.0])
    annotation = axes.texts[-1]
    assert "1000" in annotation.get_text()
    # y is in data coordinates, so it is the value itself.
    assert annotation.xy[1] == pytest.approx(1000.0)


def test_save_suppresses_the_library_version_stamp(axes, tmp_path) -> None:
    """``bnb publish --check`` compares rendered bytes, so a Matplotlib version
    written into the PNG would make an unchanged figure read as a change."""

    from PIL import Image

    axes.plot([0.0, 1.0], [0.0, 1.0])
    path = style.save(axes.figure, tmp_path / "figure.png")

    with Image.open(path) as image:
        assert "Software" not in image.info
