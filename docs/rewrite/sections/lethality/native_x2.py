"""Game v13: the real-scale set at x2 native speed. Every card and handle of the set and its
rules.toml go through the exact similarity at length 1, time 1/2 (`bnb.play.tuning`'s
`time_scaled_settings` and `time_scaled_card`, the live time scale's), except the mode
clocks (the match clock, respawn, capture, waves and the lock keep their numbers as native
seconds). Run once, from the x1 files:

    uv run --no-sync python docs/rewrite/sections/lethality/native_x2.py
"""

import dataclasses

import torch

from bnb.game import api, ships
from bnb.game.handling import NAMES
from bnb.game.handling.similarity import HANDLE_UNITS, factor
from bnb.game.ship_types import FIELDS, ShipTypes
from bnb.play.tuning import settings_of, snap_handle, time_scaled_card, time_scaled_settings

SET = "real-scale"
FACTOR = 2.0
KEPT = ("pool.", "mode.capture_seconds", "mode.wave_seconds", "mode.lock_seconds_per_root_km")


def main() -> None:
    rules = api.rules_of(api.PRESETS["real_scale_deathmatch"])
    if abs(rules.flight.gravity - 9.81) > 1e-6:
        raise SystemExit(f"the set is not at x1 (gravity {rules.flight.gravity}); run once")
    tick = rules.tick_length
    settings = settings_of(rules)
    scaled = time_scaled_settings(settings, FACTOR, tick)
    changed = {k: v for k, v in scaled.items() if v != settings[k] and not k.startswith(KEPT)}
    print("rules:", changed)
    found = ships.load(SET)
    types = ShipTypes(
        **{n: torch.tensor([s.card[n] for s in found], dtype=torch.float64) for n in FIELDS}
    )
    card = time_scaled_card(types, rules, FACTOR)
    out = []
    for i, ship in enumerate(found):
        new_card = {n: float(getattr(card, n)[i].float()) for n in FIELDS}
        handles = dict(ship.handles)
        if set(handles) == set(NAMES):
            handles = {
                n: snap_handle(n, v * factor(HANDLE_UNITS[n], 1.0, 1.0 / FACTOR), tick)
                for n, v in handles.items()
            }
        out.append(dataclasses.replace(ship, card=new_card, handles=handles))
    ships.write_rules(dict(ships.rules(SET), **changed), SET)
    for ship in out:
        ships.write_ship(ship, SET)
    print(f"{len(out)} ships scaled x{FACTOR:g}")


if __name__ == "__main__":
    main()
