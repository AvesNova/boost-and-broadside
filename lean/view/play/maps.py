"""Maps: the games the play shell starts by name, from the menu's **New match** and **Maps**
pages and from `bnb play --preset <name>`.

A map is a whole game (`Map`): its config (`GameConfig`: the mode, the arena radius, the
ships per team, the fields, the spawn groups, the clocks, the kills to win and the ship
set), its lineup (each slot's ship type by name, the same for both teams; None for the
game's own) and its tuning (`Tuning`: the rule settings and the ship cards where they
differ from the game's, and the time scale's factor; empty for most maps).

There are three places a map comes from (`Map.place`):

- "code": the game presets (`api.PRESETS`, the lineups of those with their own in
  `api.LINEUPS`), read-only: training and the tournaments play them;
- "user": the player's own, one JSON file each in the user's configuration directory
  (`USER`, like the themes), where the **Maps** page saves by default;
- "project": one JSON file each in the repository (`PROJECT`, `src/bnb/game/maps/`,
  tracked in git like the ship sets), for the maps worth keeping with the code.

A saved map's name is lowercase words joined by hyphens ("big-sky"), which no game preset
uses (theirs join words with underscores), and is unique over both places, so a name
always means one map. A file holds the config's fields (`game`; a fixed layout's fields
and spawn points under `game.layout`, `GameConfig.layout`), the lineup, and the
tuning where it is not empty: the rule settings, the time scale's factor, and the cards of
the ship types it changes by type key (`ships`), so a map stays valid when a player adds a
type to the set. The arena radius, the clocks and the kills to win are the config's own
fields: a tuning of them is folded into the config when a map is made from a match
(`Map.of_match`).
"""

import dataclasses
import json
import pathlib
import re

import platformdirs
import torch

from bnb.contracts import GameConfig
from bnb.contracts.config import layout_of
from bnb.game import api
from bnb.game.api import FLEET_PRESETS, LINEUPS, PRESETS

from .tuning import Tuning, changed_card

USER = pathlib.Path(platformdirs.user_config_dir("boost-and-broadside")) / "maps"
PROJECT = pathlib.Path(__file__).resolve().parents[1] / "game" / "maps"
NAME = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
PLACES = ("user", "project")
PLACE_LABELS = {"code": "game preset", "user": "yours", "project": "in the project"}
# The rule settings a config holds itself: a tuning of them goes into the config.
CONFIG_SETTINGS = {
    "pool.arena_radius": "arena_radius",
    "pool.episode_seconds": "episode_seconds",
    "pool.respawn_seconds": "respawn_seconds",
    "mode.kills_per_ship": "kills_per_ship",
}
# The game presets the pages offer as maps: not the fleet-size evaluation presets (the
# New match page sets the fleet size itself).
MAP_PRESETS = {name: config for name, config in PRESETS.items() if name not in FLEET_PRESETS}
MODE_WORDS = {
    "team_deathmatch": "deathmatch",
    "frontline": "frontline",
    "elimination": "elimination",
    "sandbox": "sandbox",
}
# The modes the pages offer, with their titles.
MODES = (
    ("team_deathmatch", "Team deathmatch"),
    ("frontline", "Frontline"),
    ("elimination", "Elimination"),
)


def map_of(preset: str) -> str:
    """A game preset's map key: its name without its mode's word ("deathmatch_wide" ->
    "wide", "real_scale_frontline" -> "real_scale", "frontline" -> "standard")."""
    word = MODE_WORDS[PRESETS[preset].mode]
    parts = [part for part in preset.split("_") if part != word]
    return "_".join(parts) or "standard"


RESERVED = set(PRESETS) | {map_of(name) for name in MAP_PRESETS}  # never a saved map's name


@dataclasses.dataclass(frozen=True)
class Map:
    """A game to play by name (see the module): `place` is "code", "user" or "project"."""

    name: str
    config: GameConfig
    lineup: tuple[str, ...] | None = None
    tuning: Tuning = Tuning()
    place: str = "code"

    @property
    def saved(self) -> bool:
        """Whether a player saved it (not a game preset)."""
        return self.place != "code"

    def lineup_types(self) -> tuple[int, ...] | None:
        """The lineup as type indices of the map's ship set, or None for the game's own;
        fails naming a type the set does not hold."""
        if self.lineup is None:
            return None
        names = api.type_names(self.config.ship_set)
        missing = [name for name in self.lineup if name not in names]
        if missing:
            raise ValueError(f"map {self.name}: the {self.config.ship_set} ships have no {missing}")
        return tuple(names.index(name) for name in self.lineup)

    @classmethod
    def of_match(
        cls,
        name: str,
        config: GameConfig,
        lineup: tuple[int, ...] | None,
        tuning: Tuning,
        place: str = "user",
    ) -> Map:
        """The map of a match: its config with the tuned arena radius, clocks and kills to
        win folded in (`CONFIG_SETTINGS`, the design's values at x1), its lineup by type
        name, and the rest of its tuning."""
        rules = dict(tuning.rules)
        changes = {}
        for key, field in CONFIG_SETTINGS.items():
            if key in rules:
                value = rules.pop(key)
                changes[field] = int(value) if field == "kills_per_ship" else value
        names = api.type_names(config.ship_set)
        own = api.rules_of(config).team_types
        return cls(
            name,
            dataclasses.replace(config, **changes),
            None if lineup is None or tuple(lineup) == own else tuple(names[k] for k in lineup),
            dataclasses.replace(tuning, rules=rules),
            place,
        )

    def renamed(self, name: str, place: str | None = None) -> Map:
        """The same map under another name (and place)."""
        return dataclasses.replace(self, name=name, place=self.place if place is None else place)


def preset_map(name: str) -> Map:
    """A game preset as a map (its lineup from `api.LINEUPS`)."""
    return Map(name, PRESETS[name], LINEUPS.get(name))


def to_json(game_map: Map) -> dict:
    """A saved map's file contents (see the module): the cards of the types its tuning
    changes, by type key, each with the fields that differ from the type's file."""
    data: dict = {
        "game": dataclasses.asdict(game_map.config),
        "lineup": None if game_map.lineup is None else list(game_map.lineup),
    }
    tuning, ship_set = game_map.tuning, game_map.config.ship_set
    if tuning.rules:
        data["rules"] = dict(tuning.rules)
    if tuning.card:
        card, default = tuning.types(ship_set), api.ship_types(ship_set=ship_set)
        keys = [ship.key for ship in api.ships.load(ship_set)]
        ships = {}
        for kind, key in enumerate(keys):
            fields = {
                field: float(value[kind])
                for field, value in vars(card).items()
                if not torch.equal(value[kind], getattr(default, field)[kind])
            }
            if fields:
                ships[key] = fields
        data["ships"] = ships
    if tuning.time_factor != 1.0:
        data["time_factor"] = tuning.time_factor
    return data


def from_json(name: str, data: dict, place: str) -> Map:
    """The map a file holds; fails naming any entry, config field, ship type or card field
    it does not know."""
    unknown = sorted(set(data) - {"game", "lineup", "rules", "ships", "time_factor"})
    game = dict(data["game"])
    fields = {field.name for field in dataclasses.fields(GameConfig)}
    unknown += sorted(set(game) - fields)
    unknown += sorted(set(data.get("rules", {})) - set(api.tuning.BY_KEY))
    if unknown:
        raise ValueError(f"map {name}: unknown entries {unknown}")
    game["spawn_groups"] = tuple(game.get("spawn_groups", ()))
    if game.get("layout") is not None:
        game["layout"] = layout_of(game["layout"])
    config = GameConfig(**game)
    card = {}
    ships = data.get("ships", {})
    if ships:
        keys = [ship.key for ship in api.ships.load(config.ship_set)]
        default = api.default_card(config.ship_set)
        missing = sorted(set(ships) - set(keys))
        if missing:
            raise ValueError(f"map {name}: the {config.ship_set} ships have no {missing}")
        for key, values in ships.items():
            extra = sorted(set(values) - set(default))
            if extra:
                raise ValueError(f"map {name}: {key} has unknown card fields {extra}")
        values = {
            field: tuple(
                float(ships.get(key, {}).get(field, column)) for key, column in zip(keys, columns)
            )
            for field, columns in default.items()
        }
        card = changed_card(api.ship_types(ship_set=config.ship_set, **values), config.ship_set)
    lineup = data.get("lineup")
    tuning = Tuning(
        rules={key: float(value) for key, value in data.get("rules", {}).items()},
        card=card,
        time_factor=float(data.get("time_factor", 1.0)),
    )
    return Map(name, config, None if lineup is None else tuple(lineup), tuning, place)


class Maps:
    """The maps by name: the game presets and the saved ones in `user` and `project` (the
    directories of the user's and the project's files)."""

    def __init__(
        self, user: str | pathlib.Path = USER, project: str | pathlib.Path = PROJECT
    ) -> None:
        self.directories = {"user": pathlib.Path(user), "project": pathlib.Path(project)}

    def saved(self) -> list[Map]:
        """The saved maps, the project's first, each place's by name. A file that does not
        read is left out (`problems` names it)."""
        found = []
        for place in PLACES:
            for path in self._files(place):
                try:
                    found.append(self._read(path, place))
                except ValueError, KeyError, TypeError, OSError:
                    continue
        return found

    def problems(self) -> list[str]:
        """Why each saved map file that does not read does not."""
        found = []
        for place in PLACES:
            for path in self._files(place):
                try:
                    self._read(path, place)
                except (ValueError, KeyError, TypeError, OSError) as error:
                    found.append(f"{path}: {error}")
        return found

    def names(self) -> list[str]:
        """Every name `find` takes: the game presets', then the saved maps'."""
        return [*PRESETS, *(game_map.name for game_map in self.saved())]

    def find(self, name: str) -> Map:
        """The map of this name: a game preset or a saved map; fails listing the names."""
        if name in PRESETS:
            return preset_map(name)
        place = self.place_of(name)
        if place is None:
            raise ValueError(f"no game preset or saved map {name!r}; one of {self.names()}")
        return self._read(self.path(name, place), place)

    def place_of(self, name: str) -> str | None:
        """Where a saved map of this name is ("user" or "project"), or None."""
        if NAME.fullmatch(name) is None or name in RESERVED:
            return None
        for place in PLACES:
            if self.path(name, place).is_file():
                return place
        return None

    def path(self, name: str, place: str) -> pathlib.Path:
        """Where the saved map of this name lives in `place`; fails for a name that is not
        lowercase words joined by hyphens, or that a game preset or its map uses."""
        if NAME.fullmatch(name) is None:
            raise ValueError(f"map name {name!r}: use lowercase words joined by hyphens")
        if name in RESERVED:
            raise ValueError(f"{name!r} is a game preset's name; choose another")
        return self.directories[place] / f"{name}.json"

    def save(self, game_map: Map, replacing: str | None = None) -> pathlib.Path:
        """Write a map to its place. It may replace only the saved map `replacing` (the one
        being edited, which moves when the name or the place changes); any other saved map
        of that name stays and the save fails."""
        path = self.path(game_map.name, game_map.place)
        place = self.place_of(game_map.name)
        if place is not None and game_map.name != replacing:
            raise ValueError(f"a map {game_map.name!r} is saved already ({PLACE_LABELS[place]})")
        game_map.lineup_types()  # fails for a type the set does not hold
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(to_json(game_map), indent=2) + "\n")
        if replacing is not None:  # the edited map's old file, when it moved
            for where in PLACES:
                stale = self.path(replacing, where)
                if stale != path and stale.is_file():
                    stale.unlink()
        return path

    def rename(self, name: str, new: str) -> pathlib.Path:
        """Give a saved map another name, in its place."""
        game_map = self.find(name)
        if not game_map.saved:
            raise ValueError(f"{name} is a game preset; save a copy under another name")
        if new == name:
            return self.path(name, game_map.place)
        return self.save(game_map.renamed(new), replacing=name)

    def delete(self, name: str) -> None:
        """Delete a saved map's file."""
        place = self.place_of(name)
        if place is None:
            raise ValueError(f"no saved map {name!r}")
        self.path(name, place).unlink()

    def _files(self, place: str) -> list[pathlib.Path]:
        directory = self.directories[place]
        if not directory.is_dir():
            return []
        return sorted(
            path for path in directory.glob("*.json") if NAME.fullmatch(path.stem) is not None
        )

    def _read(self, path: pathlib.Path, place: str) -> Map:
        return from_json(path.stem, json.loads(path.read_text()), place)
