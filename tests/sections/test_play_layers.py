"""The play shell's layering: it reaches the game only through `bnb.game.api` (and policy
files through `bnb.runs`), it reads no field of the game state, and rendering never
imports torch (it reads numpy snapshots)."""

import ast
import pathlib
import subprocess
import sys

PLAY = pathlib.Path(__file__).resolve().parents[2] / "src" / "bnb" / "play"
RENDER = PLAY / "render"
ALLOWED = (
    "bnb.play",
    "bnb.runtime",
    "bnb.controllers",
    "bnb.game.api",
    "bnb.contracts",
    "bnb.runs",  # policy files, for the trained agents
)


def _imported_modules(path: pathlib.Path) -> list[str]:
    """Every `bnb` module a file imports, with `from package import name` read as
    `package.name` (so `from bnb.game import api` is `bnb.game.api`)."""
    modules = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            modules += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules += [f"{node.module}.{alias.name}" for alias in node.names]
    return [module for module in modules if module.split(".")[0] == "bnb"]


def _allowed(module: str) -> bool:
    return any(module == prefix or module.startswith(prefix + ".") for prefix in ALLOWED)


def test_play_reaches_the_game_only_through_its_api():
    outside = {
        f"{path.relative_to(PLAY)}: {module}"
        for path in PLAY.rglob("*.py")
        for module in _imported_modules(path)
        if not _allowed(module)
    }
    assert not outside


def test_the_import_check_sees_a_reach_past_the_api(tmp_path):
    path = tmp_path / "reach.py"
    path.write_text("from bnb.game.physics.medium import medium\nfrom bnb.game import api\n")
    assert [_allowed(module) for module in _imported_modules(path)] == [False, True]


def test_rendering_never_imports_torch():
    """Import every render module with torch made unimportable."""
    modules = sorted(
        f"bnb.play.render.{path.stem}" for path in RENDER.glob("*.py") if path.stem != "__init__"
    )
    code = "import sys; sys.modules['torch'] = None\n" + "".join(
        f"import {module}\n" for module in modules
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


STATE_PARTS = ("ships", "bullets", "fields", "mode", "match")


def _state_reads(path: pathlib.Path) -> list[str]:
    """`state.<part>` attribute reads (the game state's fields) in a file."""
    return [
        f"{path.name}:{node.lineno} .{node.attr}"
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Attribute)
        and node.attr in STATE_PARTS
        and isinstance(node.value, ast.Name)
        and "state" in node.value.id
    ]


def test_play_reads_no_field_of_the_game_state():
    """The state is opaque outside the game: the shell reads it through `bnb.game.api`."""
    found = [read for path in PLAY.rglob("*.py") for read in _state_reads(path)]
    assert not found, "\n".join(found)


def test_the_state_check_sees_a_field_read(tmp_path):
    path = tmp_path / "reach.py"
    path.write_text("x = state.ships.x\ny = values['ships.x']\n")
    assert _state_reads(path) == ["reach.py:1 .ships"]
