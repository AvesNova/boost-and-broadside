"""An index of the names in the code, for the short names written in the docs.

Built from the source with `ast` (nothing is imported) over `src/bnb`, `bench`, `tests` and
the docs hook:

- `definitions`: every module-level class, function and assignment, and every class
  member (method, field, class attribute), by full dotted name;
- `words`: every identifier the code uses anywhere (names, attributes, arguments, keyword
  arguments) and every identifier-shaped string constant (column names, mode names,
  controller names).

A docs page may write `Rules.pools` or `GameConfig` instead of the full dotted name when it
names exactly one definition, and a lowercase word such as `shield_lost` when the code uses
it. Code that is not Python, or a plain word in backticks, opts out with `{.plain}` after
the span.
"""

import ast
import builtins
import functools
import keyword
import pathlib
import re

from .pages import ROOT

PACKAGES = (ROOT / "src" / "bnb", ROOT / "bench", ROOT / "tests", ROOT / "docs" / "hooks")
IDENTIFIER = re.compile(r"^[A-Za-z_]\w*$")


def _module_name(path: pathlib.Path) -> str:
    base = ROOT / "src" if (ROOT / "src") in path.parents else ROOT
    parts = path.relative_to(base).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _targets(node) -> list[str]:
    names = []
    for target in getattr(node, "targets", [getattr(node, "target", None)]):
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, ast.Tuple):
            names.extend(element.id for element in target.elts if isinstance(element, ast.Name))
    return names


def _instance_attributes(cls: ast.ClassDef) -> set[str]:
    """Names assigned as `self.name = ...` anywhere in a class."""
    out = set()
    for node in ast.walk(cls):
        for target in getattr(node, "targets", [getattr(node, "target", None)]):
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) \
                    and target.value.id == "self":
                out.add(target.attr)
    return out


@functools.cache
def index() -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """(definitions, words, strings): strings are the identifier-shaped string constants,
    such as environment variable names."""
    definitions, words, strings = set(), set(dir(builtins)) | set(keyword.kwlist), set()
    for package in PACKAGES:
        for path in package.rglob("*.py"):
            module = _module_name(path)
            definitions.add(module)
            words.update(module.split("."))
            tree = ast.parse(path.read_text())
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    definitions.add(f"{module}.{node.name}")
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    definitions.update(f"{module}.{name}" for name in _targets(node))
                if isinstance(node, ast.ClassDef):
                    for member in node.body:
                        if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            definitions.add(f"{module}.{node.name}.{member.name}")
                        elif isinstance(member, (ast.Assign, ast.AnnAssign)):
                            definitions.update(f"{module}.{node.name}.{name}"
                                               for name in _targets(member))
                    definitions.update(f"{module}.{node.name}.{name}"
                                       for name in _instance_attributes(node))
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    words.add(node.id)
                elif isinstance(node, ast.Attribute):
                    words.add(node.attr)
                elif isinstance(node, ast.arg):
                    words.add(node.arg)
                elif isinstance(node, ast.keyword) and node.arg:
                    words.add(node.arg)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    words.add(node.name)
                elif isinstance(node, ast.Constant) and isinstance(node.value, str) \
                        and IDENTIFIER.match(node.value):
                    words.add(node.value)
                    strings.add(node.value)
    words.update(part for name in definitions for part in name.split("."))
    return frozenset(definitions), frozenset(words), frozenset(strings)


def matches(short: str) -> list[str]:
    """Every definition whose dotted name ends with `short` (on a part boundary)."""
    definitions = index()[0]
    return sorted(name for name in definitions if name == short or name.endswith("." + short))


def is_word(word: str) -> bool:
    return word in index()[1]


def is_string(text: str) -> bool:
    """A string constant in the code, such as an environment variable's name."""
    return text in index()[2]
