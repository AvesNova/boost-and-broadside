"""The ship types, one file each: `src/bnb/game/ships/<name>.toml`, tracked in git.

`order.toml` lists the types in order: a ship type's index (the card's column, a lineup's
entry) is its place there. Each ship's file holds:

- `name`: what the interface calls it ("Fighter");
- `[handles]`: what a designer sets (`bnb.game.handling.Handles`, SI units), as the ship
  flies them: what the play shell's tuning panel edits;
- `[card]`: the physics parameters the game reads (`bnb.game.ship_types.ShipTypes`), solved
  from the handles: the default card is these, column by column.

The files are written by `write` (the tuning panel's Save), and read once at import into
`bnb.game.ship_types.DEFAULTS`; `reload` reads them again after a save.
"""

import dataclasses
import pathlib
import re
import tomllib

SHIPS = pathlib.Path(__file__).parent / "ships"
ORDER = "order.toml"
NAME = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")


@dataclasses.dataclass(frozen=True)
class Ship:
    """One ship type: its file's stem (`key`), its display name, its handles and its card
    fields (name -> value)."""

    key: str
    name: str
    handles: dict[str, float]
    card: dict[str, float]


def load(directory: str | pathlib.Path | None = None) -> list[Ship]:
    """The ship types in their order (from SHIPS by default). Fails naming the file for a
    missing ship file."""
    directory = pathlib.Path(SHIPS if directory is None else directory)
    order = tomllib.loads((directory / ORDER).read_text())["order"]
    ships = []
    for key in order:
        path = directory / f"{key}.toml"
        if not path.is_file():
            raise FileNotFoundError(f"{ORDER} lists the ship {key!r}, but {path} is missing")
        data = tomllib.loads(path.read_text())
        ships.append(
            Ship(
                key=key,
                name=data["name"],
                handles={k: float(v) for k, v in data.get("handles", {}).items()},
                card={k: float(v) for k, v in data["card"].items()},
            )
        )
    return ships


def _table(values: dict[str, float]) -> str:
    return "".join(f"{name} = {value!r}\n" for name, value in values.items())


def write(ships: list[Ship], directory: str | pathlib.Path | None = None) -> None:
    """Write the ship types (their order and one file each; to SHIPS by default), and remove
    the files of ships no longer listed."""
    directory = pathlib.Path(SHIPS if directory is None else directory)
    directory.mkdir(parents=True, exist_ok=True)
    for ship in ships:
        if NAME.fullmatch(ship.key) is None:
            raise ValueError(f"ship key {ship.key!r}: use lowercase words joined by hyphens")
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
        )
    keys = [ship.key for ship in ships]
    listed = ", ".join(f'"{key}"' for key in keys)
    (directory / ORDER).write_text(
        "# The ship types in order: a type's index is its place here (bnb.game.ships).\n"
        f"order = [{listed}]\n"
    )
    for path in directory.glob("*.toml"):
        if path.name != ORDER and path.stem not in keys:
            path.unlink()


def names(ships: list[Ship] | None = None) -> tuple[str, ...]:
    """The ship types' display names, in order."""
    return tuple(ship.name for ship in (load() if ships is None else ships))
