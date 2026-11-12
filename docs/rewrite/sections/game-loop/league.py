"""The game loop's league: the fixed references and the champions, as players
(`bench.game_loop.Player`), in `league.json` here (the champions added as rounds go)."""

import json
import math
import pathlib

from bench.game_loop import ENERGY, SHIP_SET, TURN, Player, player_from
from bnb.controllers.scripted_doctrine import DOCTRINES, from_named

HERE = pathlib.Path(__file__).parent
LEAGUE = HERE / "league.json"


def hand() -> Player:
    """Round 1's hand doctrines (`scripted_doctrines.json`)."""
    stored = json.loads(DOCTRINES.read_text())[SHIP_SET]
    return Player(
        "hand", from_named(stored[ENERGY]["doctrine"]), from_named(stored[TURN]["doctrine"])
    )


def fixed() -> list[Player]:
    """The fixed references: doctrines off (the anchor, 1000), the hand doctrines, and two
    null agents: random play, and the agent at temperature 3 (every decision flattened)."""
    return [
        Player("off"),
        hand(),
        Player("random", temperature=math.inf),
        Player("hot", temperature=3.0),
    ]


def load() -> list[Player]:
    """The league: the fixed references and every champion so far."""
    players = fixed()
    if LEAGUE.exists():
        players += [player_from(entry) for entry in json.loads(LEAGUE.read_text())]
    return players


def add(player: Player) -> None:
    """A champion joins the league."""
    entries = json.loads(LEAGUE.read_text()) if LEAGUE.exists() else []
    entries = [e for e in entries if e["name"] != player.name] + [player.named()]
    LEAGUE.write_text(json.dumps(entries, indent=1) + "\n")
