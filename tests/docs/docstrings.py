"""The docstring report: public modules, classes and functions of `bnb` and `bench` whose
docstring is missing or one word. A report, not a gate.

    uv run --no-sync python -m tests.docs.docstrings

Public means no leading underscore in the name or in any enclosing module or class. Methods
of public classes are listed; dunder methods and test code are not.
"""

import ast
import pathlib

from .pages import ROOT

PACKAGES = (ROOT / "src" / "bnb", ROOT / "bench")


def _module_name(path: pathlib.Path) -> str:
    base = ROOT / "src" if (ROOT / "src") in path.parents else ROOT
    parts = path.relative_to(base).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _thin(node) -> bool:
    docstring = ast.get_docstring(node)
    return docstring is None or len(docstring.split()) <= 1


def _public(name: str) -> bool:
    return not name.startswith("_")


def missing() -> list[tuple[str, str]]:
    """(kind, dotted name) of every public object with a missing or one-word docstring."""
    out = []
    for package in PACKAGES:
        for path in sorted(package.rglob("*.py")):
            module = _module_name(path)
            if any(not _public(part) for part in module.split(".")):
                continue
            tree = ast.parse(path.read_text())
            if _thin(tree) and path.read_text().strip():
                out.append(("module", module))
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _public(node.name):
                    if _thin(node):
                        out.append(("function", f"{module}.{node.name}"))
                elif isinstance(node, ast.ClassDef) and _public(node.name):
                    if _thin(node):
                        out.append(("class", f"{module}.{node.name}"))
                    out.extend(("method", f"{module}.{node.name}.{child.name}")
                               for child in node.body
                               if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                               and _public(child.name) and _thin(child))
    return out


def main() -> None:
    found = missing()
    print(f"docstrings: {len(found)} public object(s) with a missing or one-word docstring")
    for kind, name in found:
        print(f"  {kind:8} {name}")


if __name__ == "__main__":
    main()
