"""Yongma (the energy fighter) and Jujak (the turn fighter) as ship files of the real-scale
set: vizia's handles with the archetype deltas of `archetypes_solved.json` (the gun study's
s = 1 scaling: section 4 of docs/explanation/game/ship-design/archetypes.md) and this
section's armour (`ARMOUR`: the head-on fraction and angle and the shield's recharge, the
same on both), solved under the set's rules; the gun range follows from vizia's bullet
life at each one's cruise. A new type is appended to the set's order; an
existing one is rewritten in place.

Run: PYTHONPATH=src python docs/rewrite/sections/doctrines/ships.py
"""

import json
import pathlib
import tomllib

import torch

from bnb.game import api, ships
from bnb.game.handling import Handles

HERE = pathlib.Path(__file__).parent
SET = "real-scale"
BASE = "vizia"
SCALING = "s1"
DELTAS = HERE.parent / "gun-study" / "archetypes_solved.json"
KEYS = {"Yongma": "yongma", "Jujak": "jujak"}
# The armour both carry (handles), the tuning loop's game-design knobs; vizia's values are
# head-on fraction 0.2, angle 0.738 rad (42 degrees), recharge 12 per s after 6 s.
ARMOUR = {
    "head_on_fraction": 0.2,
    "head_on_angle": 0.737832313317803,
    "shield_recharge_rate": 12.0,
    "shield_recharge_delay": 6.0,
}
ROLE = {"Yongma": "the energy fighter", "Jujak": "the turn fighter"}


def build(name: str, rules) -> ships.Ship:
    """The ship file of `name`: vizia's handles with its delta and the armour, solved."""
    base = tomllib.loads((ships.directory_of(SET) / f"{BASE}.toml").read_text())["handles"]
    delta = json.loads(DELTAS.read_text())[SCALING]["delta"][name]
    handles = dict(base, **delta, **ARMOUR)
    # the set's bullet life (vizia's 0.6 s), so the gun range follows the cruise
    types = api.ship_types(ship_set=SET)
    index = [ship.key for ship in ships.load(SET)].index(BASE)
    cruise = torch.zeros(types.hit_damage.shape[0], dtype=torch.float32)
    cruise[index] = handles["cruise_speed"]
    handles["gun_range"] = float(api.gun_range(rules, types, cruise)[index])
    solution = api.solve_handles(
        Handles(**{n: torch.tensor([v], dtype=torch.float64) for n, v in handles.items()}),
        rules,
    )
    card = {
        field: float(getattr(solution.types, field)[0])
        for field in api.ship_types(ship_set=SET).__dataclass_fields__
    }
    notes = {}
    for handle, value in handles.items():
        if handle == "gun_range":
            notes[handle] = "follows from the set's bullet life (vizia's), cruise and muzzle speed"
        elif handle in ARMOUR:
            notes[handle] = (
                "the doctrine tuning's armour (docs/rewrite/sections/doctrines/ships.py)"
            )
        elif handle in delta:
            notes[handle] = (
                f"vizia's {base[handle]:.6g} x {value / base[handle]:.4f}: {ROLE[name]}'s "
                "archetype delta (docs/explanation/game/ship-design/archetypes.md, section 4)"
            )
        else:
            notes[handle] = "vizia's"
    return ships.Ship(key=KEYS[name], name=name, handles=handles, card=card, notes=notes)


def main() -> None:
    game = api.PRESETS["real_scale_deathmatch"]
    rules = api.rules_of(game, types=api.ship_types(ship_set=SET))
    order = ships.load(SET)
    present = {ship.key for ship in order}
    for name in KEYS:
        ship = build(name, rules)
        if ship.key in present:
            ships.write_ship(ship, SET)
        else:
            ships.append(ship, SET)
        print(name, {k: round(v, 4) for k, v in ship.handles.items() if k in ARMOUR})


if __name__ == "__main__":
    main()
