"""Whether a code reference in the docs names something that exists."""

import ast
import builtins
import dataclasses
import importlib
import importlib.util
import re
import sys

from . import names
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



SHORT = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")
FILE_NAME = re.compile(r"^[\w.-]+\.(?:md|toml|yml|yaml|lock|py|json|jsonl|npz|txt)$")


def file_name(span: str) -> str | None:
    """None if a bare file name in backticks exists somewhere in the repository."""
    if (ROOT / span).exists() or any(True for _ in ROOT.glob(f"[!.]*/**/{span}")):
        return None
    return f"no file named `{span}` in the repository; add {{.plain}} if it is generic"


def short_name(span: str) -> str | None:
    """None if a backticked name that is not a full `bnb.`/`bench.` name is fine: a lowercase
    word the code uses, a class, function or constant that names exactly one definition
    (`GameConfig`, `Rules.pools`), or a dotted name in an importable library (`torch.compile`).
    Spans that are not identifier-shaped (shapes, expressions, commands) are not checked."""
    parts = span.split(".")
    if FILE_NAME.match(span):
        return file_name(span)
    if not SHORT.match(span) or parts[0] in ("bnb", "bench"):
        return None
    if len(parts) == 1 and (span[0].islower() or span[0] == "_" or len(span) == 1):
        return None if names.is_word(span) else f"`{span}` is not a name in the code; " \
            "write it without backticks, or add {.plain} if it is code but not Python"
    if len(parts) == 1 and (span in dir(builtins) or names.is_string(span)):
        return None
    if len(parts) > 1 and parts[0] != "tests" and parts[0] not in _CLASS_LIKE() \
            and importlib.util.find_spec(parts[0]) is not None:
        return dotted_name(span)
    found = names.matches(span)
    if len(found) == 1:
        return None
    if not found:
        return f"`{span}` names nothing in src/bnb, bench or tests"
    return f"`{span}` is ambiguous; write it in full: {', '.join(found)}"


def _CLASS_LIKE() -> set[str]:
    """Names the code defines, which win over a library of the same name."""
    return {part for name in names.index()[0] for part in name.split(".")}


def _options(cls) -> set[str]:
    """A command dataclass's flags: each field's, or the name `tyro.conf.arg` gives it; a
    positional field (`tyro.conf.Positional`) has none."""
    import typing

    import tyro

    positional = typing.get_args(tyro.conf.Positional[int])[1]
    hints = typing.get_type_hints(cls, include_extras=True)
    options = set()
    for field in dataclasses.fields(cls):
        metadata = getattr(hints[field.name], "__metadata__", ())
        if positional in metadata:
            continue
        names = [getattr(meta, "name", None) for meta in metadata]
        name = next((name for name in names if name), field.name)
        if dataclasses.is_dataclass(hints[field.name]):  # a nested config: --name.field
            options |= {f"--{name}.{option[2:]}" for option in _options(hints[field.name])}
        else:
            options.add("--" + name.replace("_", "-"))
    return options


def command(arguments: str) -> str | None:
    """None if `bnb <arguments>` names a command (`bnb.cli.COMMANDS`) and uses only its
    options (`bnb.cli.Train`, `Runs`, `ShipCard`, `Ratings`, `HeadToHead`, `Crossover` or
    `Figures`, and
    `FiguresVideo` and `FiguresWatch` for `bnb figures video` and `bnb figures watch`;
    `bnb queue`'s own, `bnb.cli.QUEUE_FLAGS`, and for `bnb queue add KIND` the job's
    command's), with a known preset and mode."""
    import shlex

    from bnb.cli import (
        COMMANDS,
        QUEUE_FLAGS,
        Crossover,
        Figures,
        FiguresVideo,
        FiguresWatch,
        HeadToHead,
        Ratings,
        Runs,
        ShipCard,
        Train,
    )
    from bnb.contracts.config import MODES
    from bnb.game.modes.rules import PRESETS
    from bnb.play.app import Options as Play

    tokens = shlex.split(arguments)
    if not tokens or tokens == ["..."]:  # the command itself, or `bnb ...`
        return None
    if tokens[0] not in COMMANDS:
        return f"bnb needs a command, one of {list(COMMANDS)}"
    commands = {
        "train": Train,
        "play": Play,
        "runs": Runs,
        "ship-card": ShipCard,
        "ratings": Ratings,
        "head-to-head": HeadToHead,
        "crossover": Crossover,
        "figures": Figures,
    }
    if tokens[0] == "queue":
        kinds = {"train": Train, "resume": Train, "ratings": Ratings}
        kind = next((token for token in tokens[2:] if token in kinds), None)
        job = _options(kinds[kind]) if tokens[1:2] == ["add"] and kind else set()
        options = set(QUEUE_FLAGS) | job
    else:
        chosen = commands[tokens[0]]
        if tokens[0] == "figures" and tokens[1:2] in (["video"], ["watch"]):
            chosen = FiguresVideo if tokens[1] == "video" else FiguresWatch
        options = _options(chosen)
    values = {"--preset": set(PRESETS) | {"mixed"}, "--mode": set(MODES)}
    for index, token in enumerate(tokens):
        if not token.startswith("-"):
            continue
        flag, _, value = token.partition("=")
        if flag in ("--help", "-h"):
            continue
        if flag not in options:
            return f"bnb {tokens[0]} has no option {flag}; its options are {sorted(options)}"
        if not value and index + 1 < len(tokens):
            value = tokens[index + 1]
        if flag in values and value not in values[flag]:
            return f"bnb {flag} {value!r} is not one of {sorted(values[flag])}"
    return None
