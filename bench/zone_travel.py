"""Frontline zone-to-zone travel times: each ship type of a list flying straight (no field,
the stick centred) from two starts, coasting or holding full boost, and the time it takes
to cover each zone distance of each frontline map.

Starts: "spawn" (spawn speed and spawn energy, how a respawned ship sets out) and "cruise"
(the type's cruise speed and a full store, a ship already up to speed). Distances per map
(the ring at `zone_ring_fraction` arena radii, zone radius `zone_radius` times the map
scale): the adjacent zones' centres (a chord of 72 degrees), the spawn zone's centre to
the zone it attacks (two places on, a chord of 144 degrees: the enemy's defense zone at
front 0), and that distance less the target's radius (entering the zone).

    PYTHONPATH=src python -m bench.zone_travel [--types yongma,jujak,vizia] [--out FILE]

The slowest and fastest cruising types of the set are added as references. Prints a
Markdown table and writes one JSON document.
"""

import argparse
import json
import math

import torch

from bnb.contracts import GameConfig
from bnb.game import api, ships
from bnb.game.handling.characteristics import characteristics
from bnb.game.handling.manoeuvres import empty_map
from bnb.game.handling.manoeuvres import ships as straight_ships
from bnb.game.modes.rules import PRESETS
from bnb.game.physics.flight import airframe_of, controls, flight_tick
from bnb.game.ship_types import per_ship

SHIP_SET = "real-scale"
MAPS = (
    "real_scale_frontline",
    "real_scale_frontline_10v10_density",
    "real_scale_frontline_20v20_density",
    "real_scale_frontline_50v50_density",
)
CENTRE_STICK = 0
COAST, BOOST = 0, 1  # power indices
SECONDS = 900.0


def distances(preset: str) -> dict:
    """The map's radius, zone radius and zone distances (m)."""
    game = PRESETS[preset]
    rules = api.rules_of(game)
    mode = rules.modes[0]
    ring = mode.zone_ring_fraction * game.arena_radius
    zone = mode.zone_radius * game.arena_radius / api.MAP_SCALE_RADIUS
    attack = 2.0 * ring * math.sin(math.radians(72.0))
    return dict(
        arena_radius=game.arena_radius,
        zone_radius=zone,
        adjacent=2.0 * ring * math.sin(math.radians(36.0)),
        attack=attack,
        attack_entry=attack - zone,
    )


def flights(type_indices: list[int]) -> torch.Tensor:
    """(starts, powers, types, ticks) float64 distance flown, starts (spawn, cruise),
    powers (coast, boost)."""
    types = api.ship_types(ship_set=SHIP_SET, dtype=torch.float64)
    rules = api.rules_of(GameConfig(1, 1, api.MAP_SCALE_RADIUS, ship_set=SHIP_SET), types=types)
    card = characteristics(api.ship_types(ship_set=SHIP_SET), rules)
    count = len(type_indices)
    ship_type = torch.tensor(type_indices).repeat(4)[None]  # (1, starts x powers x types)
    start = torch.arange(2).repeat_interleave(2 * count)[None]
    power = torch.arange(2).repeat_interleave(count).repeat(2)[None]
    cruise = card.cruise_speed.double()[ship_type]
    speed = torch.where(start == 0, torch.full_like(cruise, rules.spawn_speed), cruise)
    energy = torch.where(
        start == 0,
        per_ship(types.spawn_energy, ship_type),
        per_ship(types.max_energy, ship_type),
    )
    zeros = torch.zeros_like(speed)
    ship = straight_ships(speed, energy, zeros, zeros, ship_type)
    airframe, fields = airframe_of(types, ship_type), empty_map(speed)
    stick, mode = controls(torch.where(power == 0, COAST, BOOST), torch.full_like(ship_type, 0))
    trace = []
    for _ in range(round(SECONDS / rules.tick_length)):
        ship = flight_tick(rules.flight, airframe, rules.tick_length, ship, stick, mode, fields)
        trace.append(ship.x[0].clone())
    return torch.stack(trace, -1).reshape(2, 2, count, -1), rules.tick_length, card


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--types", default="yongma,jujak,vizia")
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()
    keys = [ship.key for ship in ships.load(SHIP_SET)]
    names = api.type_names(SHIP_SET)
    chosen = [keys.index(key) for key in arguments.types.split(",")]
    types = api.ship_types(ship_set=SHIP_SET)
    rules = api.rules_of(GameConfig(1, 1, api.MAP_SCALE_RADIUS, ship_set=SHIP_SET), types=types)
    cruise = characteristics(types, rules).cruise_speed
    slowest, fastest = int(cruise.argmin()), int(cruise.argmax())
    for extra in (slowest, fastest):
        if extra not in chosen:
            chosen.append(extra)
    trace, tick, card = flights(chosen)
    maps = {preset: distances(preset) for preset in MAPS}
    rows = []
    print(
        "| map | R (m) | zone r (m) | type | cruise / top (m/s) | distance | m | "
        "spawn coast | spawn boost | cruise coast | cruise boost |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for preset, geometry in maps.items():
        for column, index in enumerate(chosen):
            for kind in ("adjacent", "attack", "attack_entry"):
                metres = geometry[kind]
                times = {}
                for start, start_name in enumerate(("spawn", "cruise")):
                    for power, power_name in enumerate(("coast", "boost")):
                        reached = (trace[start, power, column] >= metres).nonzero()
                        seconds = float(reached[0, 0] + 1) * tick if len(reached) else math.inf
                        times[f"{start_name}_{power_name}"] = seconds
                row = dict(
                    map=preset,
                    arena_radius=geometry["arena_radius"],
                    zone_radius=geometry["zone_radius"],
                    type=names[index],
                    cruise_speed=float(card.cruise_speed[index]),
                    top_speed=float(card.top_speed[index]),
                    distance=kind,
                    metres=metres,
                    **times,
                )
                rows.append(row)
                print(
                    f"| {preset} | {geometry['arena_radius']:.0f} | {geometry['zone_radius']:.0f}"
                    f" | {names[index]} | {row['cruise_speed']:.0f} / {row['top_speed']:.0f} | "
                    f"{kind} | {metres:.0f} | "
                    + " | ".join(f"{times[k]:.1f}" for k in times)
                    + " |"
                )
    if arguments.out:
        with open(arguments.out, "w") as file:
            json.dump(rows, file, indent=1)


if __name__ == "__main__":
    main()
