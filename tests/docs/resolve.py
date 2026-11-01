"""Whether a code reference in the docs names something that exists."""

import ast
import dataclasses
import importlib
import sys

from .pages import ROOT

if str(ROOT) not in sys.path:  # `bench` is a top-level package of the repository
    sys.path.insert(0, str(ROOT))


def _attribute(owner, name: str):
    """owner.name, also for dataclass fields and annotated attributes without a default."""
    if hasattr(owner, name):
        return getattr(owner, name)
    if dataclasses.is_dataclass(owner) and name in {f.name for f in dataclasses.fields(owner)}:
        return None
    if name in getattr(owner, "__annotations__", {}):
        return None
    raise AttributeError(name)


def dotted_name(name: str) -> str | None:
    """None if `name` (a.b.c) imports, else why not: the longest importable module prefix,
    then attributes."""
    parts = name.split(".")
    for split in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:split]))
        except ModuleNotFoundError:
            continue
        for index, attribute in enumerate(parts[split:], split):
            try:
                obj = _attribute(obj, attribute)
            except AttributeError:
                return f"{'.'.join(parts[:index])} has no attribute {attribute!r}"
            if obj is None and index + 1 < len(parts):
                return f"cannot look inside {'.'.join(parts[:index + 1])}"
        return None
    return f"no module {parts[0]!r}"


def repository_path(reference: str) -> str | None:
    """None if the path exists (and, for `path::name`, the file defines `name`)."""
    path, _, test = reference.partition("::")
    target = ROOT / path
    if not target.exists():
        return f"{path} does not exist"
    if test:
        names = {node.name for node in ast.walk(ast.parse(target.read_text()))
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        if test not in names:
            return f"{path} defines no {test!r}"
    return None


def reference(text: str) -> str | None:
    if "/" in text:
        return repository_path(text)
    return dotted_name(text)


def python_imports(code: str) -> list[tuple[int, str]]:
    """(line in the block, dotted name) of every `bnb` or `bench` import in a Python block."""
    out = []
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if node.module.split(".")[0] in ("bnb", "bench"):
                out.extend((node.lineno, f"{node.module}.{alias.name}") for alias in node.names)
        elif isinstance(node, ast.Import):
            out.extend((node.lineno, alias.name) for alias in node.names
                       if alias.name.split(".")[0] in ("bnb", "bench"))
    return out

