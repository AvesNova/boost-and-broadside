"""Imports only point down the layer order of docs/rewrite/scaffold.md section 4."""

import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "bnb"
LAYER = {"contracts": 0, "game": 1, "controllers": 2, "models": 2, "learn": 3, "runtime": 4,
         "league": 5, "play": 6, "cli": 7}


def _imports(path: pathlib.Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module)
        elif isinstance(node, ast.Import):
            out.update(alias.name for alias in node.names)
    return {m for m in out if m.startswith("bnb.")}


def test_imports_point_down():
    bad = []
    for path in SRC.rglob("*.py"):
        package = path.relative_to(SRC).parts[0].removesuffix(".py")
        if package == "__init__":
            continue
        for module in _imports(path):
            target = module.split(".")[1]
            if LAYER[target] > LAYER[package]:
                bad.append(f"{path.relative_to(SRC)} imports {module}")
    assert not bad, "\n".join(bad)
