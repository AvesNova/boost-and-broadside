"""The coverage map, as warnings: a page whose covered code changed after it did shows up
in pytest's warning summary. It never fails the suite."""

import warnings

import pytest

from . import staleness


class PossiblyStalePage(UserWarning):
    """A page's covered code changed in commits after the page's last change."""


def test_pages_newer_than_the_code_they_cover():
    reason = staleness.unavailable()
    if reason:
        pytest.skip(f"no git history to compare: {reason}")
    for stale in staleness.report():
        warnings.warn(staleness.describe(stale), PossiblyStalePage, stacklevel=1)
