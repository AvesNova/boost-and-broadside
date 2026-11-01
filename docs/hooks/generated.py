"""MkDocs hook: content generated from the code and the data files at build time.

A page asks for a block with a line of its own, outside a fenced block:

    <!-- generated: ship-columns -->

The hook replaces the line with Markdown built from `bnb`, `pyproject.toml`,
`docs/roadmap.yml` or `docs/packages.yml`. An unknown name fails the build. `BLOCKS` lists
every name; `render(name, page)` is what the tests call.

It also turns every backticked dotted `bnb.` or `bench.` name in the prose into a link to
the reference, so readers can click through and the strict build checks each one.
"""

import ast
import dataclasses
import inspect
import io
import pathlib
import posixpath
import re
import sys
import tokenize
import tomllib
from collections.abc import Callable

ROOT = pathlib.Path(__file__).resolve().parents[2]
ROADMAP = ROOT / "docs" / "roadmap.yml"
PACKAGES = ROOT / "docs" / "packages.yml"
MARKER = re.compile(r"^<!-- generated: ([a-z0-9-]+) -->[ \t]*$", re.MULTILINE)
FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
# A backticked dotted name not already inside a link: `bnb.x.y` -> [`bnb.x.y`][bnb.x.y].
CODE_NAME = re.compile(r"(?<![\[`])`((?:bnb|bench)(?:\.\w+)+)`(?![`\]{])")
# A spec key in a source comment, such as [cl_max]: the spec is not part of the site.
SPEC_KEY = re.compile(r"\s*\[[^\]]*\]")


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
    """Line number to the text of the comment on that line, without spec keys."""
    text = pathlib.Path(source_file).read_text()
    out = {}
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.COMMENT:
            out[token.start[0]] = SPEC_KEY.sub("", token.string.lstrip("#")).strip(" ,;")
    return out


def _link(page: str, target: str, text: str) -> str:
    """A Markdown link from `page` to `target`, both relative to docs/."""
    return f"[{text}]({posixpath.relpath(target, posixpath.dirname(page) or '.')})"


def _yaml(path: pathlib.Path):
    import yaml

    return yaml.safe_load(path.read_text())


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
    """The observation's ship columns, and which are zero for a hidden ship."""
    from bnb.contracts.obs import SHIP_COLUMNS, SHIP_HIDDEN_ZERO

    rows = [(index, _code(name), "zero" if index < SHIP_HIDDEN_ZERO else "shown")
            for index, name in enumerate(SHIP_COLUMNS)]
    return _table(("Index", "Column", "For a ship the team cannot see"), rows)


def _columns(names: tuple[str, ...]) -> str:
    return _table(("Index", "Column"), [(index, _code(name)) for index, name in enumerate(names)])


def field_columns() -> str:
    """The observation's field columns."""
    from bnb.contracts.obs import FIELD_COLUMNS

    return _columns(FIELD_COLUMNS)


def zone_columns() -> str:
    """The observation's zone columns."""
    from bnb.contracts.obs import ZONE_COLUMNS

    return _columns(ZONE_COLUMNS)


def global_columns() -> str:
    """The observation's global columns."""
    from bnb.contracts.obs import GLOBAL_COLUMNS

    return _columns(GLOBAL_COLUMNS)


# ------------------------------------------------------------------ actions and rewards

def action_layout() -> str:
    """The action components and their sizes, from bnb.contracts.action."""
    from bnb.contracts import action

    rows = [(_code("power"), _code("NUM_POWER"), action.NUM_POWER),
            (_code("stick"), _code("NUM_STICK"), action.NUM_STICK),
            (_code("shoot"), _code("NUM_SHOOT"), action.NUM_SHOOT),
            ("joint", _code("NUM_JOINT"), action.NUM_JOINT)]
    return _table(("Component", "Constant", "Choices"), rows)


def stick_levels() -> str:
    """The stick level of each stick index."""
    from bnb.game.rules import STICK_LEVELS

    return _table(("Stick index", "Level (positive turns left)"),
                  [(index, _number(level)) for index, level in enumerate(STICK_LEVELS)])


def power_modes() -> str:
    """The modes with their names, from the comment on POWER_MODES in bnb.game.rules."""
    from bnb.game import rules
    from bnb.game.rules import POWER_MODES

    line = next(number for number, text in
                enumerate(pathlib.Path(inspect.getsourcefile(rules)).read_text().splitlines(), 1)
                if text.startswith("POWER_MODES ="))
    names = _comments(inspect.getsourcefile(rules))[line].split(":")[-1].split(",")
    if len(names) != len(POWER_MODES):
        raise ValueError("the comment on POWER_MODES must name every mode")
    return _table(("Power index", "Name", "Power mode"),
                  [(index, name.strip(), _number(mode))
                   for index, (name, mode) in enumerate(zip(names, POWER_MODES))])


def reward_levels() -> str:
    """The reward levels in order, with their default weights."""
    from bnb.contracts import REWARD_LEVELS
    from bnb.learn.rewards import RewardConfig

    weights = RewardConfig().weights
    return _table(("Index", "Level", "Default weight"),
                  [(index, _code(level), _number(weight))
                   for index, (level, weight) in enumerate(zip(REWARD_LEVELS, weights))])


def reward_config() -> str:
    """The fields of RewardConfig."""
    from bnb.learn.rewards import RewardConfig

    return _dataclass_table(RewardConfig)


# ------------------------------------------------------------------ the game

def ship_card() -> str:
    """The ship card's default parameters by ship type."""
    from bnb.game import ship_types

    comments = _keyword_comments(ship_types, "DEFAULTS")
    rows = [(_code(name), _number(fighter), _number(heavy), comments.get(name, ""))
            for name, (fighter, heavy) in ship_types.DEFAULTS.items()]
    return _table(("Parameter", "Fighter (type 0)", "Heavy (type 1)", "Note in the source"),
                  rows)


def game_constants() -> str:
    """The shared constants of Rules."""
    from bnb.game.rules import Rules

    # The sizes and the games come from the GameConfig; team_types defaults to slots
    # alternating fighter and heavy; the flight constants have their own table.
    return _dataclass_table(Rules, skip=("ships_per_team", "num_fields", "pools", "team_types",
                                          "flight"))


def flight_constants() -> str:
    """The flight constants every ship type shares."""
    from bnb.game.rules import FlightConstants

    return _dataclass_table(FlightConstants)


def mode_clocks() -> str:
    """Each mode's default episode length and respawn wait."""
    from bnb.game.modes.rules import CLOCKS

    return _table(("Mode", "Episode (s)", "Respawn wait (s)"),
                  [(_code(mode), _number(episode), _number(respawn))
                   for mode, (episode, respawn) in CLOCKS.items()])


def mode_rules() -> str:
    """The fields of ModeRules."""
    from bnb.game.modes.rules import ModeRules

    return _dataclass_table(ModeRules)


def game_presets() -> str:
    """The training game presets."""
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
    """The fields of GameConfig."""
    from bnb.contracts import GameConfig

    return _dataclass_table(GameConfig)


def pool_fields() -> str:
    """The fields of Pool."""
    from bnb.contracts import Pool

    return _dataclass_table(Pool)


# ------------------------------------------------------------------ training

def training_mix() -> str:
    """The default training mix: each game preset's share."""
    from bnb.runtime.train import TRAINING_MIX

    return _table(("Game preset", "Share of the batch"),
                  [(_code(name), _number(share)) for name, share in TRAINING_MIX])


def training_lineup() -> str:
    """The skeleton lineup's pools."""
    from bnb.runtime.train import LINEUP

    return _table(("Pool", "Share", "Seats (role 0, role 1)", "Fog share"),
                  [(_code(pool.name), _number(pool.share), ", ".join(pool.seats),
                    _number(pool.fog_share)) for pool in LINEUP])


def mixed_pools() -> str:
    """The pools of the mixed preset, and the batch size multiple."""
    from bnb.contracts.pools import environment_multiple
    from bnb.runtime.train import mixed

    _, pools = mixed()
    rows = [(_code(pool.name), _code(pool.mode), _number(pool.arena_radius),
             _number(pool.share), ", ".join(pool.seats)) for pool in pools]
    table = _table(("Pool", "Mode", "Arena radius (m)", "Share", "Seats"), rows)
    return f"{table}\n\nBatch size multiple: {environment_multiple(pools)} environments."


def train_config() -> str:
    """The fields of TrainConfig."""
    from bnb.runtime.train import TrainConfig

    return _dataclass_table(TrainConfig, skip=("pools", "learner"))


def learner_config() -> str:
    """The fields of LearnerConfig."""
    from bnb.learn.ppo import LearnerConfig

    return _dataclass_table(LearnerConfig)


def cli_options() -> str:
    """The options of the bnb command."""
    from bnb.cli import Train

    rows = []
    for field in dataclasses.fields(Train):
        flag = "--" + field.name.replace("_", "-")
        rows.append((_code(flag), _code(_number(field.default))))
    return _table(("Option", "Default"), rows)


def play_options() -> str:
    """The options of python -m bnb.play."""
    from bnb.play.app import Options

    comments = _field_comments(Options)
    rows = [(_code("--" + field.name.replace("_", "-")), _code(_number(field.default)),
             comments.get(field.name, "")) for field in dataclasses.fields(Options)]
    return _table(("Option", "Default", "Note in the source"), rows)


def play_keys() -> str:
    """The play shell's keys."""
    from bnb.play.render.hud import KEYS

    return _table(("Key", "Does"), [(key, meaning) for key, meaning in KEYS])


def play_themes() -> str:
    """The shipped themes, and the values each theme file lists."""
    from bnb.play import themes

    names = ", ".join(_code(name) + (" (the default)" if name == themes.DEFAULT else "")
                      for name in themes.shipped_names())
    labels = [(_code(field.name), field.metadata["label"])
              for field in dataclasses.fields(themes.Theme) if field.name != "name"]
    return f"Themes: {names}.\n\n" + _table(("Value", "In the theme editor"), labels)


def yemong_config() -> str:
    """The fields of YemongConfig."""
    from bnb.models.yemong.model import YemongConfig

    return _dataclass_table(YemongConfig)


def bench_sizes() -> str:
    """The ranges and budget of the benchmark's random sizes."""
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

ROADMAP_STATUSES = ("merged", "milestone 1 built", "specified", "decided", "planned")
ROADMAP_KEYS = {"section", "title", "status", "spec", "page"}


def roadmap_entries() -> list[dict]:
    """docs/roadmap.yml, checked: every entry has the keys, a known status, a spec that
    exists (or none) and a page that exists."""
    entries = _yaml(ROADMAP)
    for entry in entries:
        if set(entry) != ROADMAP_KEYS:
            raise ValueError(f"roadmap entry {entry} needs exactly {sorted(ROADMAP_KEYS)}")
        if entry["status"] not in ROADMAP_STATUSES:
            raise ValueError(f"roadmap status {entry['status']!r} is not one of "
                             f"{ROADMAP_STATUSES}")
        if entry["spec"] and not (ROOT / entry["spec"]).exists():
            raise ValueError(f"roadmap spec {entry['spec']} does not exist")
        if not (ROOT / "docs" / entry["page"]).exists():
            raise ValueError(f"roadmap page {entry['page']} does not exist")
    return entries


def _section(number: str) -> dict:
    found = [entry for entry in roadmap_entries() if entry["section"] == number]
    if len(found) != 1:
        raise ValueError(f"the roadmap has no single section {number!r}")
    return found[0]


def roadmap(page: str) -> str:
    """The roadmap table: every section, its status, record and page."""
    rows = [(entry["section"], _link(page, entry["page"], entry["title"]), entry["status"],
             _code(entry["spec"]) if entry["spec"] else "none yet")
            for entry in roadmap_entries()]
    return _table(("Section", "Title", "Status", "Design record"), rows)


def roadmap_summary(page: str) -> str:
    """One line per status: the sections that have it, linked to their pages."""
    lines = []
    for status in ROADMAP_STATUSES:
        entries = [entry for entry in roadmap_entries() if entry["status"] == status]
        if entries:
            names = ", ".join(_link(page, entry["page"], f"{entry['title']} ({entry['section']})")
                              for entry in entries)
            lines.append(f"- **{status.capitalize()}:** {names}.")
    return "\n".join(lines)


STATUS_SENTENCES = {
    "merged": "is merged",
    "milestone 1 built": "has its first milestone built; the later milestones are to follow",
    "specified": "is specified and not built",
    "decided": "has its decisions recorded; its spec is to follow",
    "planned": "is planned",
}


def section_status(page: str) -> str:
    """The status of the section whose page this is, from the roadmap."""
    found = [entry for entry in roadmap_entries() if entry["page"] == page]
    if len(found) != 1:
        raise ValueError(f"no single roadmap section names {page} as its page")
    entry = found[0]
    record = f" Its design record is `{entry['spec']}`." if entry["spec"] else ""
    return (f"*Status:* section {entry['section']} ({entry['title']}) "
            f"{STATUS_SENTENCES[entry['status']]}.{record}")


def roadmap_in_progress(page: str) -> str:
    """The sections not merged yet, linked to their in-progress pages."""
    return "\n".join(f"- {_link(page, entry['page'], entry['title'])} (section "
                     f"{entry['section']}, {entry['status']})"
                     for entry in roadmap_entries() if entry["status"] != "merged")


# ------------------------------------------------------------------ packages and versions

def _layers() -> dict[str, int]:
    """LAYER from the import-direction test, the source of truth for the layers."""
    tree = ast.parse((ROOT / "tests" / "contract" / "test_imports.py").read_text())
    for node in tree.body:
        names = [target.id for target in getattr(node, "targets", ())
                 if isinstance(target, ast.Name)]
        if isinstance(node, ast.Assign) and "LAYER" in names:
            return ast.literal_eval(node.value)
    raise ValueError("no LAYER in tests/contract/test_imports.py")


def package_entries() -> list[dict]:
    """docs/packages.yml, checked against the layers and the tree: every layered package is
    listed, a package that exists names its reference page, and one that does not names its
    roadmap section."""
    entries = _yaml(PACKAGES)
    layers = _layers()
    listed = {entry["package"] for entry in entries}
    if set(layers) - listed:
        raise ValueError(f"docs/packages.yml is missing {sorted(set(layers) - listed)}")
    for entry in entries:
        name = entry["package"]
        exists = name == "bench" and (ROOT / "bench").is_dir() or \
            (ROOT / "src" / "bnb" / name).is_dir() or (ROOT / "src" / "bnb" / f"{name}.py").exists()
        if exists and not (ROOT / "docs" / entry.get("reference", "")).is_file():
            raise ValueError(f"package {name} exists but has no reference page in packages.yml")
        if not exists and "section" not in entry:
            raise ValueError(f"package {name} does not exist and names no roadmap section")
        if name != "bench" and name not in layers:
            raise ValueError(f"package {name} has no layer in tests/contract/test_imports.py")
    return entries


def packages(page: str) -> str:
    """The packages, their layers and where each is documented or planned."""
    layers = _layers()
    rows = []
    for entry in package_entries():
        name = entry["package"]
        package = _code(name if name == "bench" else f"bnb.{name}")
        if "reference" in entry:
            where = _link(page, entry["reference"], "reference")
        else:
            section = _section(entry["section"])
            where = (f"not built: section {section['section']}, "
                     f"{_link(page, section['page'], section['status'])}")
        rows.append((layers.get(name, "outside"), package, entry["holds"], where))
    return _table(("Layer", "Package", "Holds", "Where"), rows)


def versions() -> str:
    """The Python and dependency requirements, from pyproject.toml."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    uv = project["tool"]["uv"]
    index = {entry["name"]: entry["url"] for entry in uv.get("index", [])}
    sources = uv.get("sources", {})
    pinned = (ROOT / ".python-version").read_text().strip()
    rows = [("Python", _code(project["project"]["requires-python"]),
             f"uv; `.python-version` names {pinned}")]
    for requirement in project["project"]["dependencies"]:
        name = re.split(r"[<>=!~ \[]", requirement, maxsplit=1)[0]
        source = sources.get(name, {}).get("index")
        rows.append((name, _code(requirement[len(name):] or "any"),
                     f"the `{source}` index, {index[source]}" if source else "PyPI"))
    groups = ", ".join(_code(group) for group in uv.get("default-groups", ()))
    table = _table(("Requirement", "Version", "From"), rows)
    return f"{table}\n\nDevelopment groups installed by default: {groups}."


# ------------------------------------------------------------------ the hook

BLOCKS: dict[str, Callable[..., str]] = {
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
    "learner-config": learner_config,
    "cli-options": cli_options,
    "yemong-config": yemong_config,
    "play-options": play_options,
    "play-keys": play_keys,
    "play-themes": play_themes,
    "bench-sizes": bench_sizes,
    "roadmap": roadmap,
    "roadmap-summary": roadmap_summary,
    "roadmap-in-progress": roadmap_in_progress,
    "section-status": section_status,
    "packages": packages,
    "versions": versions,
    "block-names": lambda: block_names(),
}


def block_names() -> str:
    """Every block, with the first line of its function's docstring where it has one."""
    lines = []
    for name, function in BLOCKS.items():
        if name == "block-names":
            continue
        summary = (inspect.getdoc(function) or "").split("\n")[0].rstrip(".")
        lines.append(f"- `{name}`" + (f": {summary[0].lower()}{summary[1:]}."
                                        if summary else ""))
    return "\n".join(lines)


def render(name: str, page: str = "index.md") -> str:
    """The block `name` as it appears on `page` (relative to docs/; links depend on it)."""
    if name not in BLOCKS:
        raise ValueError(f"unknown generated block {name!r}; known: {sorted(BLOCKS)}")
    function = BLOCKS[name]
    if "page" in inspect.signature(function).parameters:
        return function(page)
    return function()


def expand(markdown: str, page: str = "index.md") -> str:
    """The page with every marker line outside fenced blocks replaced by its block, and every
    backticked dotted name outside fenced blocks linked to the reference."""
    out, fence = [], None
    for line in markdown.split("\n"):
        opening = FENCE.match(line)
        if fence is None and opening:
            fence = opening.group(1)
            out.append(line)
            continue
        if fence is not None:
            if line.strip().startswith(fence):
                fence = None
            out.append(line)
            continue
        marker = MARKER.match(line)
        if marker is None:
            out.append(CODE_NAME.sub(r"[`\1`][\1]", line))
            continue
        try:
            out.append(render(marker.group(1), page))
        except ValueError as error:
            raise ValueError(f"{page}: {error}") from error
    return "\n".join(out)


def on_page_markdown(markdown: str, page, config, files) -> str:
    return expand(markdown, page.file.src_path)
