"""Source-walking checks of the design rules (code style, Layers): the game state is opaque
to the runtime, and process-wide settings and compiles stay where they belong."""

import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "bnb"
RUNTIME = SRC / "runtime"

# The state's parts (bnb.game.state.State). `ships` and `mode` are also names of other
# things (Obs.ships, GameConfig.mode), so those two count only on a state expression.
STATE_ONLY_PARTS = ("match", "bullets")
STATE_PARTS = ("ships", "mode")
# What the runtime may import from the game: the API, and the offline handling tools.
GAME_MODULES = ("bnb.game", "bnb.game.api", "bnb.game.handling.characteristics")


def _sources(root: pathlib.Path):
    for path in sorted(root.rglob("*.py")):
        yield path, ast.parse(path.read_text())


def _names_state(node: ast.expr) -> bool:
    """The expression is a state: a name containing `state`, or `x["state"]`."""
    if isinstance(node, ast.Name):
        return "state" in node.id
    if isinstance(node, ast.Attribute):
        return "state" in node.attr
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
        return node.slice.value == "state"
    return False


def test_the_runtime_reads_no_field_of_the_game_state():
    found = []
    for path, tree in _sources(RUNTIME):
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and (
                node.attr in STATE_ONLY_PARTS
                or (node.attr in STATE_PARTS and _names_state(node.value))
            ):
                found.append(f"{path.relative_to(SRC)}:{node.lineno} .{node.attr}")
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                if node.module.startswith("bnb.game") and node.module not in GAME_MODULES:
                    found.append(f"{path.relative_to(SRC)}:{node.lineno} imports {node.module}")
    assert not found, "\n".join(found)


# Process-wide settings: calls on `torch` (or its submodules) and assignments to these.
SETTING_CALLS = {
    "torch.manual_seed",
    "torch.set_float32_matmul_precision",
    "torch.set_num_threads",
    "torch.set_num_interop_threads",
    "torch.use_deterministic_algorithms",
    "torch.cuda.manual_seed",
    "torch.cuda.manual_seed_all",
}
SETTING_ASSIGNMENTS = ("torch.backends.", "torch._dynamo.config.", "torch._inductor.config.")


def _dotted(node: ast.expr) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


# a spawned child process's target: its process's entry point (the play shell's work aside)
CHILD_ENTRY_POINTS = {
    ("game/handling/characteristics.py", "_measure_child"),
    ("runtime/realtime.py", "_build"),
}


def _entry_point(path: pathlib.Path, function: str | None) -> bool:
    """`bnb.cli`, a package's `__main__`, a module's `main` function, or a child process's
    target (`CHILD_ENTRY_POINTS`)."""
    return (
        path == SRC / "cli.py"
        or path.name == "__main__.py"
        or function == "main"
        or (path.relative_to(SRC).as_posix(), function) in CHILD_ENTRY_POINTS
    )


def _calls_and_assignments(tree: ast.AST):
    """(function name or None, node) for every call and assignment target."""

    def visit(node, function):
        for child in ast.iter_child_nodes(node):
            inner = child.name if isinstance(child, ast.FunctionDef) else function
            if isinstance(child, ast.Call):
                yield inner, child.func
            if isinstance(child, (ast.Assign, ast.AugAssign)):
                targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                for target in targets:
                    yield inner, target
            yield from visit(child, inner)

    yield from visit(tree, None)


def test_process_wide_settings_belong_to_the_entry_points():
    found = []
    for path, tree in _sources(SRC):
        for function, node in _calls_and_assignments(tree):
            name = _dotted(node)
            setting = name in SETTING_CALLS or name.startswith(SETTING_ASSIGNMENTS)
            if setting and not _entry_point(path, function):
                found.append(f"{path.relative_to(SRC)}:{node.lineno} {name}")
    assert not found, "\n".join(found)


def test_only_the_runtime_compiles():
    """Section code does not compile itself: the runtime chooses the compiled stages."""
    found = []
    for path, tree in _sources(SRC):
        for function, node in _calls_and_assignments(tree):
            if _dotted(node) == "torch.compile" and not (
                RUNTIME in path.parents or _entry_point(path, function)
            ):
                found.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert not found, "\n".join(found)
