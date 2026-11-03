"""The code is formatted by `ruff format` and passes `ruff check` (code style, Tooling)."""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _ruff(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ruff", *arguments], cwd=ROOT, capture_output=True, text=True
    )


def test_the_code_is_formatted():
    result = _ruff("format", "--check", "--force-exclude", "src", "bench", "tests")
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_code_passes_the_linter():
    result = _ruff("check", "src", "bench", "tests")
    assert result.returncode == 0, result.stdout + result.stderr
