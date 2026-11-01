"""The play shell's layering: rendering never imports torch (it reads numpy snapshots)."""

import pathlib
import subprocess
import sys

RENDER = pathlib.Path(__file__).resolve().parents[2] / "src" / "bnb" / "play" / "render"


def test_rendering_never_imports_torch():
    """Import every render module with torch made unimportable."""
    modules = sorted(f"bnb.play.render.{path.stem}" for path in RENDER.glob("*.py")
                     if path.stem != "__init__")
    code = ("import sys; sys.modules['torch'] = None\n"
            + "".join(f"import {module}\n" for module in modules))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

