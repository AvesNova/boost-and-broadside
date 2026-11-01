"""Every example marked `<!-- runnable -->` runs, on the CPU, from the repository root.

Python blocks run in a fresh interpreter; command blocks in bash. CUDA is hidden from both,
so a runnable example cannot use the GPU by accident. Examples that need a GPU, the
benchmark lock or minutes are marked `<!-- not-runnable: reason -->` instead.
"""

import os
import subprocess
import sys

import pytest

from . import pages

pytestmark = pytest.mark.slow

TIMEOUT_S = 300
RUNNABLE = [fence for fence in pages.fences() if fence.runnable]


def _environment() -> dict[str, str]:
    return dict(os.environ, CUDA_VISIBLE_DEVICES="", NO_MKDOCS_2_WARNING="1")


@pytest.mark.parametrize("fence", RUNNABLE, ids=lambda fence: fence.location)
def test_runnable_example_runs(fence):
    if fence.language == "python":
        command = [sys.executable, "-c", fence.code]
    else:
        command = ["bash", "-euo", "pipefail", "-c", fence.code]
    done = subprocess.run(command, cwd=pages.ROOT, env=_environment(), capture_output=True,
                          text=True, timeout=TIMEOUT_S)
    assert done.returncode == 0, (f"{fence.location}: exit {done.returncode}\n"
                                  f"{done.stdout[-2000:]}\n{done.stderr[-4000:]}")
