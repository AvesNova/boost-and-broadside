"""The race correctness interface over bnb.game, eager on the CPU in float32.

States travel as numpy dicts under the race's flat key names (`bnb.game.state.CANONICAL`).
The observation is translated from the contract layout back to the race's: rows in sim
order, `ally` from team_rel, times in ticks, no team_rel / since_seen / spawned columns.
Bullets are not observed (contract decision 2026-10-31).
"""

import dataclasses

import numpy as np
import torch
from race_presets import preset as race_preset

from bnb.contracts import Action, GameConfig
from bnb.contracts.obs import SHIP
from bnb.game import match
from bnb.game.physics import flight as flight_mod
from bnb.game.physics.bullets import bullet_motion
from bnb.game.physics.medium import medium
from bnb.game.rules import FlightConstants, Rules
from bnb.game.ship_types import DEFAULTS, ship_types
from bnb.game.state import Bullets, Fields, Ships, State, blank, flat, unflat

RACE_SHIP = ("x", "y", "c", "s", "vx", "vy", "w", "alpha", "r", "E", "shield", "alive", "visible",
             "ally", "gun_wait", "respawn", "type")
TYPES = ship_types()


def rules(spec) -> Rules:
    p = race_preset(spec)
    return match.rules_of(GameConfig(p["num_ships"] // 2, p["num_fields"], p["arena_radius"]))


def to_torch(arrays: dict) -> dict:
    return {k: torch.as_tensor(np.asarray(v)) for k, v in arrays.items()}


def to_numpy(tensors: dict) -> dict:
    return {k: v.detach().cpu().numpy() for k, v in tensors.items()}


def fields_of(f: dict) -> Fields:
    f = to_torch(f)
    return Fields(x=f["fx"], y=f["fy"], radius=f["fr"], width=f["fw"], level=f["flevel"])


def state_of(p: Rules, arrays: dict) -> State:
    """A race state dict as a State; `last_seen`, absent from race states, starts at zero."""
    st = to_torch(arrays)
    if "last_seen" not in st:
        st["last_seen"] = blank(p, st["x"].shape[0]).match.last_seen
    return unflat(st)


def ships_of(arrays: dict) -> Ships:
    """Ships with the given flight keys; every other field zero (ship_type fighter)."""
    st = to_torch(arrays)
    zeros = torch.zeros_like(st["x"])
    return Ships(**{f.name: st.get(f.name, zeros if f.name != "ship_type" else zeros.int())
                    for f in dataclasses.fields(Ships)})


def race_observation(p: Rules, st: State) -> dict:
    _, obs, truth = match.perceive(p, st)
    B, T, N, _ = obs.ships.shape
    inverse = torch.argsort(truth.view_order[0], dim=-1)  # (T, N): row of each sim ship
    views = torch.arange(T)[:, None]
    ships = obs.ships[:, views, inverse]
    cols = []
    for name in RACE_SHIP:
        if name == "ally":
            cols.append((ships[..., SHIP["team_rel"]] == 0).float())
        elif name in ("gun_wait", "respawn"):
            cols.append(torch.round(ships[..., SHIP[name]] / p.tick))
        else:
            cols.append(ships[..., SHIP[name]])
    return dict(ships=torch.stack(cols, -1), fields=obs.fields[:, 0],
                globals=obs.globals[:, 0, :2])


def _flat(st: State) -> dict:
    out = flat(st)
    del out["last_seen"]
    return out


class RaceArm:
    name = "bnb"

    def fields(self, fields, x, y):
        out = medium(fields_of(fields), torch.as_tensor(x), torch.as_tensor(y))
        return tuple(v.numpy() for v in out)

    def flight_tick(self, ship, power, stick, fields, flight=None):
        """flight: overrides applied to every ship type (shared or ship-card parameters)."""
        flight = dict(flight or {})
        shared = {k: flight.pop(k) for k in list(flight) if k not in DEFAULTS}
        F = dataclasses.replace(FlightConstants(), **shared)
        types = ship_types(**flight)
        sigma, pi = flight_mod.controls(torch.as_tensor(power), torch.as_tensor(stick))
        ships = ships_of({k: ship[k] for k in ship if k in Ships.__dataclass_fields__})
        A = flight_mod.airframe(types, ships.ship_type)
        out = flight_mod.flight_tick(F, A, Rules.tick, ships, sigma, pi, fields_of(fields))
        return {k: getattr(out, k).numpy() for k in ("x", "y", "c", "s", "w", "alpha", "ca", "sa",
                                                     "r", "E", "logn", "gx", "gy")}

    def bullet_tick(self, bullets, fields):
        b = to_torch(bullets)
        b = Bullets(**{k: b["b" + k] for k in ("x", "y", "dx", "dy", "w", "logn", "gx", "gy")},
                    ttl=torch.ones_like(b["bx"], dtype=torch.int32))
        out = bullet_motion(rules("squad"), b, fields_of(fields))
        return {"b" + k: getattr(out, k).numpy() for k in ("x", "y", "dx", "dy", "w", "logn", "gx", "gy")}

    def reset(self, preset, uniforms):
        p = rules(preset)
        return to_numpy(_flat(match.reset(p, torch.as_tensor(uniforms), TYPES)))

    def step(self, preset, state, actions, spread, uniforms):
        p = rules(preset)
        st, ev = match.step(p, state_of(p, state), Action.unpack(torch.as_tensor(actions)),
                            torch.as_tensor(spread), torch.as_tensor(uniforms), TYPES)
        events = {k: getattr(ev, k) for k in ("damage", "boundary_damage", "died", "killer", "fired", "done")}
        return to_numpy(_flat(st)), to_numpy(events), to_numpy(race_observation(p, st))

    def observe(self, preset, state):
        p = rules(preset)
        return to_numpy(race_observation(p, state_of(p, state)))

    def chaser(self, preset, state):
        from race_chaser import chaser

        p = rules(preset)
        return chaser(p, state_of(p, state)).numpy()
