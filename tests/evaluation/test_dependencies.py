"""Dependency direction and required architecture-input guards."""

import ast
import inspect
from pathlib import Path

from boost_and_broadside.models.yemong.policy import YemongPolicy
from boost_and_broadside.train.rl.policy_io import build_policy
from boost_and_broadside.train.rl.roster import EloRoster

_PACKAGE = Path(__file__).parents[2] / "src" / "boost_and_broadside"


def _module_name(path: Path) -> str:
    return ".".join(path.relative_to(_PACKAGE.parent).with_suffix("").parts)


def test_user_facing_modes_do_not_import_other_user_facing_modes():
    mode_files = sorted((_PACKAGE / "modes").glob("*.py"))
    parsed = {path: ast.parse(path.read_text()) for path in mode_files}
    user_facing = {
        _module_name(path)
        for path, tree in parsed.items()
        if any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("run_")
            for node in tree.body
        )
    }
    violations = []
    for path, tree in parsed.items():
        if _module_name(path) not in user_facing:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in user_facing:
                violations.append(f"{path.name}:{node.lineno} imports {node.module}")
            elif isinstance(node, ast.Import):
                violations.extend(
                    f"{path.name}:{node.lineno} imports {alias.name}"
                    for alias in node.names
                    if alias.name in user_facing
                )
    assert violations == []


def test_package_library_never_exits_the_process():
    violations = []
    for path in sorted(_PACKAGE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "sys"
                and node.func.attr == "exit"
            ):
                violations.append(f"{path.name}:{node.lineno}")
    assert violations == []


def test_team_pma_indices_are_required_at_policy_and_roster_construction_seams():
    policy_parameter = inspect.signature(YemongPolicy).parameters["team_pma_k"]
    builder_parameter = inspect.signature(build_policy).parameters["team_pma_k"]
    assert policy_parameter.default is inspect.Parameter.empty
    assert builder_parameter.default is inspect.Parameter.empty
    roster_parameter = inspect.signature(EloRoster.load_policy).parameters["team_pma_k"]
    assert roster_parameter.default is inspect.Parameter.empty


def test_every_policy_constructor_call_passes_team_pma_indices_explicitly():
    violations = []
    for root in (_PACKAGE, Path(__file__).parents[1]):
        for path in root.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.Call):
                    continue
                name = node.func.id if isinstance(node.func, ast.Name) else None
                if name not in {"YemongPolicy", "build_policy"}:
                    continue
                if not any(keyword.arg == "team_pma_k" for keyword in node.keywords):
                    violations.append(f"{path}:{node.lineno}")
    assert violations == []
