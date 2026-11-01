"""The site builds in strict mode: broken links, missing pages, pages left out of the nav,
bad mkdocstrings targets and unknown generated tables all fail.

Needs the `docs` dependency group, which `uv sync` installs by default; without it, the
test skips and says so.
"""

import os
import subprocess
import sys

import pytest

from . import pages

pytestmark = pytest.mark.slow


def test_the_site_builds_in_strict_mode(tmp_path):
    reason = "the docs build needs the docs dependency group: uv sync"
    pytest.importorskip("mkdocs", reason=reason)
    pytest.importorskip("material", reason=reason)
    done = subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict", "--site-dir", str(tmp_path)],
        cwd=pages.ROOT, env=dict(os.environ, NO_MKDOCS_2_WARNING="1", CUDA_VISIBLE_DEVICES=""),
        capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-8000:]
