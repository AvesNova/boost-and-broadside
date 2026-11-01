"""MkDocs hook: tables generated from the code at build time.

A page asks for a table with a line of its own:

    <!-- generated-table: ship-columns -->

The hook replaces the line with a Markdown table built from `bnb` (or from
`docs/roadmap.yml`). An unknown table name fails the build. `TABLES` lists every name;
`render(name)` is what the tests call.
"""

import ast
import dataclasses
import inspect
import io
import pathlib
import re
import sys
import tokenize
from collections.abc import Callable

ROOT = pathlib.Path(__file__).resolve().parents[2]
ROADMAP = ROOT / "docs" / "roadmap.yml"
MARKER = re.compile(r"^<!-- generated-table: ([a-z0-9-]+) -->[ \t]*$", re.MULTILINE)
FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


# ------------------------------------------------------------------ helpers

def _table(header: tuple[str, ...], rows: list[tuple]) -> str:
    """A Markdown table; every cell is rendered with str()."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def _code(value) -> str:
    return f"`{value}`"


def _number(value) -> str:
    """Integers as they are; floats to six significant digits."""
    if isinstance(value, bool) or not isinstance(value, float):
        return str(value)
    return f"{value:.6g}"


def _comments(source_file: str) -> dict[int, str]:
    """Line number to the text of the comment on that line."""
    text = pathlib.Path(source_file).read_text()
    out = {}
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.COMMENT:
            out[token.start[0]] = token.string.lstrip("#").strip()
    return out


def _field_comments(cls: type) -> dict[str, str]:
    """A dataclass's field name to the comment on its line in the source."""
    source_file = inspect.getsourcefile(cls)
    comments = _comments(source_file)
    tree = ast.parse(pathlib.Path(source_file).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
            return {statement.target.id: comments.get(statement.lineno, "")
                    for statement in node.body
                    if isinstance(statement, ast.AnnAssign)
                    and isinstance(statement.target, ast.Name)}
    raise ValueError(f"no class {cls.__name__} in {source_file}")


def _keyword_comments(module, name: str) -> dict[str, str]:
    """For `name = dict(key=..., ...)` in a module: key to the comment on its line."""
    source_file = inspect.getsourcefile(module)
    comments = _comments(source_file)
    tree = ast.parse(pathlib.Path(source_file).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                                                and target.id == name for target in node.targets):
            return {keyword.arg: comments.get(keyword.value.end_lineno, "")
                    for keyword in node.value.keywords}
    raise ValueError(f"no assignment to {name} in {source_file}")


def _dataclass_table(cls: type, skip: tuple[str, ...] = ()) -> str:
    """Every field of a dataclass: its default (or "required") and its source comment."""
    comments = _field_comments(cls)
    rows = []
    for field in dataclasses.fields(cls):
        if field.name in skip:
            continue
        if field.default is not dataclasses.MISSING:
            default = _code(_number(field.default))
        elif field.default_factory is not dataclasses.MISSING:
            default = _code(_number(field.default_factory()))
        else:
            default = "required"
        rows.append((_code(field.name), default, comments.get(field.name, "")))
    if not any(row[2] for row in rows):
        return _table(("Field", "Default"), [row[:2] for row in rows])
    return _table(("Field", "Default", "Note in the source"), rows)


# ------------------------------------------------------------------ the observation

def ship_columns() -> str:
    from bnb.contracts.obs import SHIP_COLUMNS, SHIP_HIDDEN_ZERO

    rows = [(index, _code(name), "zero" if index < SHIP_HIDDEN_ZERO else "shown")
            for index, name in enumerate(SHIP_COLUMNS)]
    return _table(("Index", "Column", "For a ship the team cannot see"), rows)


def _columns(names: tuple[str, ...]) -> str:
    return _table(("Index", "Column"), [(index, _code(name)) for index, name in enumerate(names)])


def field_columns() -> str:
    from bnb.contracts.obs import FIELD_COLUMNS

    return _columns(FIELD_COLUMNS)


def zone_columns() -> str:
    from bnb.contracts.obs import ZONE_COLUMNS

    return _columns(ZONE_COLUMNS)


def global_columns() -> str:
    from bnb.contracts.obs import GLOBAL_COLUMNS

    return _columns(GLOBAL_COLUMNS)


# ------------------------------------------------------------------ actions and rewards

def action_layout() -> str:
    from bnb.contracts import action

    rows = [(_code("power"), _code("NUM_POWER"), action.NUM_POWER),
            (_code("stick"), _code("NUM_STICK"), action.NUM_STICK),
            (_code("shoot"), _code("NUM_SHOOT"), action.NUM_SHOOT),
            ("joint", _code("NUM_JOINT"), action.NUM_JOINT)]
    return _table(("Component", "Constant", "Choices"), rows)


def stick_levels() -> str:
    from bnb.game.rules import STICK_LEVELS

    return _table(("Stick index", "Level (positive turns left)"),
                  [(index, _number(level)) for index, level in enumerate(STICK_LEVELS)])


def power_modes() -> str:
    from bnb.game.rules import POWER_MODES

    return _table(("Power index", "Power mode"),
                  [(index, _number(mode)) for index, mode in enumerate(POWER_MODES)])


def reward_levels() -> str:
    from bnb.contracts import REWARD_LEVELS
    from bnb.learn.rewards import RewardConfig

    weights = RewardConfig().weights
    return _table(("Index", "Level", "Default weight"),
                  [(index, _code(level), _number(weight))
                   for index, (level, weight) in enumerate(zip(REWARD_LEVELS, weights))])


def reward_config() -> str:
    from bnb.learn.rewards import RewardConfig

    return _dataclass_table(RewardConfig)


# ------------------------------------------------------------------ the game

def ship_card() -> str:
    from bnb.game import ship_types

    comments = _keyword_comments(ship_types, "DEFAULTS")
    rows = [(_code(name), _number(fighter), _number(heavy), comments.get(name, ""))
            for name, (fighter, heavy) in ship_types.DEFAULTS.items()]
    return _table(("Parameter", "Fighter (type 0)", "Heavy (type 1)", "Note in the source"),
                  rows)


def game_constants() -> str:
    from bnb.game.rules import Rules

    return _dataclass_table(Rules, skip=("flight",))


def flight_constants() -> str:
    from bnb.game.rules import FlightConstants

    return _dataclass_table(FlightConstants)


def mode_clocks() -> str:
    from bnb.game.modes.rules import CLOCKS

    return _table(("Mode", "Episode (s)", "Respawn wait (s)"),
                  [(_code(mode), _number(episode), _number(respawn))
                   for mode, (episode, respawn) in CLOCKS.items()])


def mode_rules() -> str:
    from bnb.game.modes.rules import ModeRules

    return _dataclass_table(ModeRules)


def game_presets() -> str:
    from bnb.game.modes.rules import CLOCKS, PRESETS

    rows = []
    for name, game in PRESETS.items():
        episode, respawn = CLOCKS[game.mode]
        rows.append((_code(name), _code(game.mode), game.ships_per_team, game.num_fields,
                     _number(game.arena_radius), _code(game.spawn_groups) if game.spawn_groups
                     else "uniform", _number(game.episode_s if game.episode_s else episode),
                     _number(game.respawn_s if game.respawn_s else respawn)))
    return _table(("Preset", "Mode", "Ships per team", "Fields", "Arena radius (m)",
                   "Spawn groups", "Episode (s)", "Respawn wait (s)"), rows)


def game_config() -> str:
    from bnb.contracts import GameConfig

    return _dataclass_table(GameConfig)


def pool_fields() -> str:
    from bnb.contracts import Pool

    return _dataclass_table(Pool)


# ------------------------------------------------------------------ training

def training_mix() -> str:
    from bnb.runtime.train import TRAINING_MIX

    return _table(("Game preset", "Share of the batch"),
                  [(_code(name), _number(share)) for name, share in TRAINING_MIX])


def training_lineup() -> str:
    from bnb.runtime.train import LINEUP

    return _table(("Pool", "Share", "Seats (role 0, role 1)", "Fog share"),
                  [(_code(pool.name), _number(pool.share), ", ".join(pool.seats),
                    _number(pool.fog_share)) for pool in LINEUP])


def mixed_pools() -> str:
    from bnb.contracts.pools import environment_multiple
    from bnb.runtime.train import mixed

    _, pools = mixed()
    rows = [(_code(pool.name), _code(pool.mode), _number(pool.arena_radius),
             _number(pool.share), ", ".join(pool.seats)) for pool in pools]
    table = _table(("Pool", "Mode", "Arena radius (m)", "Share", "Seats"), rows)
    return f"{table}\n\nBatch size multiple: {environment_multiple(pools)} environments."


def train_config() -> str:
    from bnb.runtime.train import TrainConfig

    return _dataclass_table(TrainConfig, skip=("pools", "ppo"))


def ppo_config() -> str:
    from bnb.learn.ppo import PPOConfig

    return _dataclass_table(PPOConfig)


def cli_options() -> str:
    from bnb.cli import Train

    rows = []
    for field in dataclasses.fields(Train):
        flag = "--" + field.name.replace("_", "-")
        rows.append((_code(flag), _code(_number(field.default))))
    return _table(("Option", "Default"), rows)


def yemong_config() -> str:
    from bnb.models.yemong.model import YemongConfig

    return _dataclass_table(YemongConfig)


def bench_sizes() -> str:
    # `bench` is a top-level package of the repository, not of the installed project. MkDocs
    # restores sys.path after loading a hook, so the path is added at call time.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from bench import sizes

    rows = [(_code("SHIPS"), _code(sizes.SHIPS), "ships per environment, even, inclusive"),
            (_code("FIELDS"), _code(sizes.FIELDS), "fields per environment, inclusive"),
            (_code("DENSITY"), _code(sizes.DENSITY),
             "arena radius factor on 2600 sqrt(N / 10)"),
            (_code("SHIP_BUDGET"), _code(sizes.SHIP_BUDGET), "environments x ships per batch"),
            (_code("SKELETON_SEED"), _code(sizes.SKELETON_SEED), "the public seed")]
    return _table(("Constant", "Value", "Meaning"), rows)


# ------------------------------------------------------------------ the roadmap

ROADMAP_STATUSES = ("merged", "specified", "decided", "planned")


def roadmap_entries() -> list[dict]:
    """docs/roadmap.yml, checked: every entry has the four keys, a known status, and a
    spec that exists (or none)."""
    import yaml

    entries = yaml.safe_load(ROADMAP.read_text())
    for entry in entries:
        if set(entry) != {"section", "title", "status", "spec"}:
            raise ValueError(f"roadmap entry {entry} needs section, title, status and spec")
        if entry["status"] not in ROADMAP_STATUSES:
            raise ValueError(f"roadmap status {entry['status']!r} is not one of "
                             f"{ROADMAP_STATUSES}")
        if entry["spec"] and not (ROOT / entry["spec"]).exists():
            raise ValueError(f"roadmap spec {entry['spec']} does not exist")
    return entries


def roadmap() -> str:
    rows = [(entry["section"], entry["title"], entry["status"],
             _code(entry["spec"]) if entry["spec"] else "none yet")
            for entry in roadmap_entries()]
    return _table(("Section", "Title", "Status", "Design record"), rows)


# ------------------------------------------------------------------ the hook

TABLES: dict[str, Callable[[], str]] = {
    "ship-columns": ship_columns,
    "field-columns": field_columns,
    "zone-columns": zone_columns,
    "global-columns": global_columns,
    "action-layout": action_layout,
    "stick-levels": stick_levels,
    "power-modes": power_modes,
    "reward-levels": reward_levels,
    "reward-config": reward_config,
    "ship-card": ship_card,
    "game-constants": game_constants,
    "flight-constants": flight_constants,
    "mode-clocks": mode_clocks,
    "mode-rules": mode_rules,
    "game-presets": game_presets,
    "game-config": game_config,
    "pool-fields": pool_fields,
    "training-mix": training_mix,
    "training-lineup": training_lineup,
    "mixed-pools": mixed_pools,
    "train-config": train_config,
    "ppo-config": ppo_config,
    "cli-options": cli_options,
    "yemong-config": yemong_config,
    "bench-sizes": bench_sizes,
    "roadmap": roadmap,
}


def render(name: str) -> str:
    if name not in TABLES:
        raise ValueError(f"unknown generated table {name!r}; known: {sorted(TABLES)}")
    return TABLES[name]()


def expand(markdown: str, where: str = "") -> str:
    """The page with every marker line outside fenced blocks replaced by its table."""
    out, fence = [], None
    for line in markdown.split("\n"):
        opening = FENCE.match(line)
        if fence is None and opening:
            fence = opening.group(1)
        elif fence is not None and line.strip().startswith(fence):
            fence = None
        marker = MARKER.match(line) if fence is None else None
        if marker is None:
            out.append(line)
            continue
        try:
            out.append(render(marker.group(1)))
        except ValueError as error:
            raise ValueError(f"{where}: {error}") from error
    return "\n".join(out)


def on_page_markdown(markdown: str, page, config, files) -> str:
    return expand(markdown, page.file.src_path)
