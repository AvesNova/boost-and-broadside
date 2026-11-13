"""The ship types, one file each, tracked in git, in ship sets.

A ship set is a directory of ship files: the standard set is `src/bnb/game/ships/` itself,
and every other set a subdirectory of it named after the set (`ships/real-scale/`). A game
names its set (`GameConfig.ship_set`). In each set, `order.toml` lists the types in order: a
ship type's index (the card's column, a lineup's entry) is its place there. A set may hold
`rules.toml`, the rule values its cards were solved under (`rules`). Each ship's file holds:

- `name`: what the interface calls it ("Fighter");
- `[handles]`: what a designer sets (`bnb.game.handling.Handles`, SI units), as the ship
  flies them: what the play shell's tuning panel edits;
- `[card]`: the physics parameters the game reads (`bnb.game.ship_types.ShipTypes`), solved
  from the handles: the default card is these, column by column.

The files are written by `write` (the tuning panel's Save) and `append` (its Add), and
read when a set's default card is first asked for (`bnb.game.ship_types.defaults`);
`reload` reads them again after a save.
"""

import dataclasses
import functools
import json
import pathlib
import re
import tomllib

SHIPS = pathlib.Path(__file__).parent / "ships"  # the standard set; other sets inside it
STANDARD = "standard"
ORDER = "order.toml"
RULES = "rules.toml"
NAME = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
# The tables of a set's rules.toml and the owners of their keys (`bnb.game.tuning`'s naming):
# [rules] Rules fields, [flight] FlightConstants, [mode] ModeRules.
RULE_TABLES = ("rules", "flight", "mode")


def directory_of(ship_set: str = STANDARD) -> pathlib.Path:
    """The directory of a ship set's files. Fails for a name that is not lowercase words
    joined by hyphens."""
    if NAME.fullmatch(ship_set) is None:
        raise ValueError(f"ship set {ship_set!r}: use lowercase words joined by hyphens")
    return SHIPS if ship_set == STANDARD else SHIPS / ship_set


def sets() -> tuple[str, ...]:
    """The ship sets: the standard one, then every subdirectory with an order, by name."""
    found = sorted(path.parent.name for path in SHIPS.glob(f"*/{ORDER}"))
    return (STANDARD, *found)


@functools.cache
def rules(ship_set: str = STANDARD) -> dict[str, float]:
    """A set's rule values by `bnb.game.tuning`-style key (`rules.<field>`,
    `flight.<field>`, `mode.<field>`), from its rules.toml; empty for a set without one
    (the standard set: the game's own rules). Read once; `rules.cache_clear()` forgets it."""
    path = directory_of(ship_set) / RULES
    if not path.is_file():
        return {}
    data = tomllib.loads(path.read_text())
    unknown = sorted(set(data) - set(RULE_TABLES))
    if unknown:
        raise ValueError(f"{path}: unknown tables {unknown}; use {list(RULE_TABLES)}")
    return {
        f"{table}.{name}": value
        for table in RULE_TABLES
        for name, value in data.get(table, {}).items()
    }


@dataclasses.dataclass(frozen=True)
class Ship:
    """One ship type: its file's stem (`key`), its display name, its handles and its card
    fields (name -> value), and notes on its handles (name -> text: where a value comes
    from, or that it is a placeholder), which the files keep through a save."""

    key: str
    name: str
    handles: dict[str, float]
    card: dict[str, float]
    notes: dict[str, str] = dataclasses.field(default_factory=dict)


def write_rules(values: dict[str, float], ship_set: str = STANDARD) -> None:
    """Write a set's rules.toml from values by key (as `rules` returns them); no values
    removes it (the set then plays the game's own rules)."""
    path = directory_of(ship_set) / RULES
    rules.cache_clear()
    if not values:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    unknown = sorted(key for key in values if key.split(".", 1)[0] not in RULE_TABLES)
    if unknown:
        raise ValueError(f"rule keys {unknown}: use owners {list(RULE_TABLES)}")
    tables = [
        f"[{table}]\n"
        + _table(
            {
                key.split(".", 1)[1]: value
                for key, value in values.items()
                if key.startswith(f"{table}.")
            }
        )
        for table in RULE_TABLES
        if any(key.startswith(f"{table}.") for key in values)
    ]
    path.write_text(
        "# The rule values this set's cards were solved under (bnb.game.ships.rules); the\n"
        "# game's own rules elsewhere. Times in s, lengths in m, per unit mass.\n\n"
        + "\n".join(tables)
    )


def load(ship_set: str = STANDARD) -> list[Ship]:
    """A set's ship types in their order. Fails naming the file for a missing ship file."""
    directory = directory_of(ship_set)
    order = tomllib.loads((directory / ORDER).read_text())["order"]
    ships = []
    for key in order:
        path = directory / f"{key}.toml"
        if not path.is_file():
            raise FileNotFoundError(f"{ORDER} lists the ship {key!r}, but {path} is missing")
        ships.append(read(path, key))
    return ships


def read(path: pathlib.Path, key: str) -> Ship:
    """One ship file."""
    data = tomllib.loads(path.read_text())
    return Ship(
        key=key,
        name=data["name"],
        handles={k: float(v) for k, v in data.get("handles", {}).items()},
        card={k: float(v) for k, v in data["card"].items()},
        notes={k: str(v) for k, v in data.get("notes", {}).items()},
    )


def _table(values: dict[str, float | str]) -> str:
    """TOML lines: numbers as Python writes them, text as a basic string, a bool as TOML's
    true or false."""
    return "".join(
        f"{name} = {json.dumps(value) if isinstance(value, (str, bool)) else repr(value)}\n"
        for name, value in values.items()
    )


def write(ships: list[Ship], ship_set: str = STANDARD) -> None:
    """Write a set's ship types (their order and one file each), and remove the files of
    ships no longer listed."""
    directory = directory_of(ship_set)
    for ship in ships:
        write_ship(ship, ship_set)
    keys = [ship.key for ship in ships]
    _write_order(keys, directory)
    for path in directory.glob("*.toml"):
        if path.name not in (ORDER, RULES) and path.stem not in keys:
            path.unlink()


def append(ship: Ship, ship_set: str = STANDARD) -> None:
    """Add a ship type to a set: its file, and its key last in the set's order (the other
    types' files are left as they are). Fails for a key the set has already."""
    directory = directory_of(ship_set)
    keys = tomllib.loads((directory / ORDER).read_text())["order"]
    if ship.key in keys:
        raise ValueError(f"the {ship_set} set has a ship {ship.key!r} already")
    write_ship(ship, ship_set)
    _write_order([*keys, ship.key], directory)


def _write_order(keys: list[str], directory: pathlib.Path) -> None:
    listed = ", ".join(f'"{key}"' for key in keys)
    (directory / ORDER).write_text(
        "# The ship types in order: a type's index is its place here (bnb.game.ships).\n"
        f"order = [{listed}]\n"
    )


def write_ship(ship: Ship, ship_set: str = STANDARD) -> None:
    """Write one ship type's file (not the set's order)."""
    if NAME.fullmatch(ship.key) is None:
        raise ValueError(f"ship key {ship.key!r}: use lowercase words joined by hyphens")
    directory = directory_of(ship_set)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{ship.key}.toml").write_text(
        f"# {ship.name}: a ship type (bnb.game.ships). The tuning panel in `bnb play` saves\n"
        "# it; [card] is solved from [handles], so edit the handles, not the card.\n"
        f"name = {ship.name!r}\n\n"
        "[handles]\n"
        "# What a designer sets (bnb.game.handling.Handles, SI units), as the ship flies.\n"
        f"{_table(ship.handles)}\n"
        "[card]\n"
        "# The physics parameters the game reads (bnb.game.ship_types.ShipTypes).\n"
        f"{_table(ship.card)}"
        + (
            "\n[notes]\n# Where each handle's value comes from.\n" + _table(ship.notes)
            if ship.notes
            else ""
        )
    )


def names(ship_set: str = STANDARD) -> tuple[str, ...]:
    """A set's ship types' display names, in order."""
    return tuple(ship.name for ship in load(ship_set))
