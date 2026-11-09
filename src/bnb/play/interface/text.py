"""Names and numbers as the interface writes them: ships by type and slot, clocks, and
the ship card's types (their names from the ship files, `bnb.game.api.type_names`)."""

import numpy as np

from bnb.game import api

TEAM_NAMES = ("Team 0", "Team 1")


def type_name(ship_type: int) -> str:
    """A ship type's name, from its ship file ("Type 7" for a type without one)."""
    names = api.type_names()
    return names[ship_type] if 0 <= ship_type < len(names) else f"Type {ship_type}"


def slot_of(team: np.ndarray, index: int) -> int:
    """A ship's slot in its team, from its sim index and every ship's team (sim order)."""
    return int((team[:index] == team[index]).sum())


def ship_name(team: np.ndarray, ship_type: np.ndarray, index: int) -> str:
    """A ship by its type and its slot counted from 1 ("Heavy 2"); its team shows by
    colour, or by `ship_title`."""
    return f"{type_name(int(ship_type[index]))} {slot_of(team, index) + 1}"


def ship_title(team: np.ndarray, ship_type: np.ndarray, index: int) -> str:
    """A ship with its team ("Team 1 Heavy 2")."""
    return f"{TEAM_NAMES[int(team[index])]} {ship_name(team, ship_type, index)}"


def clock(seconds: float) -> str:
    """Minutes and seconds ("2:05")."""
    whole = max(int(seconds), 0)
    return f"{whole // 60}:{whole % 60:02d}"
