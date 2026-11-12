"""How far each ship type of a set goes outward when it spawns flying straight at a wall and
turns away at once: from spawn speed and spawn energy, heading +x, the stick full over, in
each power mode, the largest +x reached in 30 s. The set's largest is what
`Rules.spawn_turn_distance` covers (`bnb.game.modes.spawns._heading`).

Usage: `python bench/spawn_turn_excursion.py [ship_set=standard]`
Prints a line per type, then one JSON line.
"""

import json
import sys

import torch

from bnb.contracts import GameConfig
from bnb.game import api
from bnb.game.handling.manoeuvres import empty_map, ships
from bnb.game.physics.flight import airframe_of, controls, flight_tick
from bnb.game.ship_types import per_ship

SECONDS = 30.0
FULL_STICK = 6  # the stick index of a full turn
POWERS = ("coast", "boost", "reverse")  # by power index


def measure(ship_set: str) -> dict:
    """Each type's largest outward reach from spawn in this set (the module docstring)."""
    types = api.ship_types(ship_set=ship_set, dtype=torch.float64)
    rules = api.rules_of(GameConfig(1, 1, api.MAP_SCALE_RADIUS, ship_set=ship_set), types=types)
    count = types.max_energy.shape[0]
    ship_type = torch.arange(count).repeat_interleave(len(POWERS))[None]
    power = torch.arange(len(POWERS)).repeat(count)[None]
    speed = torch.full(ship_type.shape, rules.spawn_speed, dtype=torch.float64)
    zeros = torch.zeros_like(speed)
    ship = ships(speed, per_ship(types.spawn_energy, ship_type), zeros, zeros, ship_type)
    airframe, fields = airframe_of(types, ship_type), empty_map(speed)
    stick, mode = controls(power, torch.full_like(ship_type, FULL_STICK))
    farthest = torch.zeros_like(speed)
    for _ in range(round(SECONDS / rules.tick_length)):
        ship = flight_tick(rules.flight, airframe, rules.tick_length, ship, stick, mode, fields)
        farthest = torch.maximum(farthest, ship.x)
    by_type = farthest.reshape(count, len(POWERS))
    names = api.type_names(ship_set)
    for index, name in enumerate(names):
        print(
            f"{name:16s} "
            + " ".join(
                f"{power} {float(by_type[index, i]):6.1f} m" for i, power in enumerate(POWERS)
            )
        )
    return dict(
        ship_set=ship_set,
        spawn_speed=rules.spawn_speed,
        largest=float(farthest.max()),
        largest_type=names[int(by_type.amax(-1).argmax())],
        spawn_turn_distance=rules.spawn_turn_distance,
    )


if __name__ == "__main__":
    arguments = dict(argument.split("=", 1) for argument in sys.argv[1:])
    print(json.dumps(measure(arguments.get("ship_set", "standard"))))
