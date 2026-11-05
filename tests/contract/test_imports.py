"""Imports only point down the layer order of docs/rewrite/scaffold.md section 4."""

import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "bnb"
LAYER = {
    "contracts": 0,
    "game": 1,
    "controllers": 2,
    "models": 2,
    "learn": 3,
    "runtime": 4,
    "runs": 5,
    "league": 6,
    "play": 7,
    "figures": 7,
    "cli": 8,
}


def _module(path: pathlib.Path) -> str:
    parts = ("bnb",) + path.relative_to(SRC).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _absolute(path: pathlib.Path, node: ast.ImportFrom) -> str:
    """The absolute module an import names, a relative one (`from ..x import y`) resolved
    against the importing module's package."""
    if node.level == 0:
        return node.module
    package = _module(path).split(".")
    if path.name != "__init__.py":
        package = package[:-1]
    base = package[: len(package) - (node.level - 1)]
    return ".".join(base + ([node.module] if node.module else []))


def _imports(path: pathlib.Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            out.add(_absolute(path, node))
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
