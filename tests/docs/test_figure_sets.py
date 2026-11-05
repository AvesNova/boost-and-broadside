"""The figures page (`docs/guides/figures.md`) describes every figure set, and only those:
one `## \`name\`` section per entry of `bnb.figures.sets.FIGURE_SETS`."""

import re

from bnb.figures.sets import FIGURE_SETS

from . import pages

PAGE = pages.DOCS / "guides" / "figures.md"


def described() -> list[str]:
    """The figure sets the page has a section for, in page order."""
    return re.findall(r"^## `([^`]+)`\s*$", PAGE.read_text(), flags=re.MULTILINE)


def test_the_figures_page_has_one_section_per_figure_set():
    sections = described()
    assert len(sections) == len(set(sections)), f"a set is described twice: {sections}"
    missing = sorted(set(FIGURE_SETS) - set(sections))
    unknown = sorted(set(sections) - set(FIGURE_SETS))
    assert not missing, f"{PAGE.name} does not describe the figure sets {missing}"
    assert not unknown, f"{PAGE.name} describes figure sets that do not exist: {unknown}"
