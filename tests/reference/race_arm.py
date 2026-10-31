"""The race correctness interface over bnb.game, eager on the CPU in float32.

The observation is translated from the contract layout back to the race's: rows in sim
order, `ally` from team_rel, times in ticks, no team_rel / since_seen / spawned columns.
"""

import numpy as np
import torch

from bnb.contracts.obs import BULLET, SHIP
from bnb.game import match
from bnb.game.perception.observe import perceive
from bnb.game.physics import flight as flight_mod
from bnb.game.physics.bullets import BULLET_MOTION_KEYS, bullet_motion
from bnb.game.physics.constants import PRESETS, preset_of, type_flights
from bnb.game.physics.medium import medium
from bnb.game.physics.ship_types import flight_of
from bnb.game.physics.step import step as physics_step

RACE_SHIP = ("x", "y", "c", "s", "vx", "vy", "w", "alpha", "r", "E", "shield", "alive", "visible",
             "ally", "gun_wait", "respawn", "type")


def to_torch(arrays: dict) -> dict:
    return {k: torch.as_tensor(np.asarray(v)) for k, v in arrays.items()}


def to_numpy(tensors: dict) -> dict:
    return {k: v.detach().cpu().numpy() for k, v in tensors.items()}


def with_memory(st: dict) -> dict:
    """Race states have no `last_seen`; start it at zero."""
    if "last_seen" not in st:
        B, N = st["x"].shape
        st = dict(st, last_seen=torch.zeros((B, 2, N), dtype=torch.int32))
    return st


def race_observation(p, st: dict) -> dict:
    _, obs, truth = perceive(p, with_memory(st))
    B, T, N, _ = obs.ships.shape
    K = p.slots
    inverse = torch.argsort(truth.view_order[0], dim=-1)  # (T, N): row of each sim ship
    views = torch.arange(T)[:, None]
    ships = obs.ships[:, views, inverse]
    cols = []
    for name in RACE_SHIP:
        if name == "ally":
            cols.append((ships[..., SHIP["team_rel"]] == 0).float())
        elif name in ("gun_wait", "respawn"):
            cols.append(torch.round(ships[..., SHIP[name]] / p.h))
        else:
            cols.append(ships[..., SHIP[name]])
    binv = (inverse[..., None] * K + torch.arange(K)).reshape(T, N * K)
    bullets = obs.bullets[:, views, binv][..., [BULLET[c] for c in ("x", "y", "vx", "vy", "visible")]]
    return dict(ships=torch.stack(cols, -1), bullets=bullets, fields=obs.fields[:, 0],
                globals=obs.globals[:, 0, :2])


class RaceArm:
    name = "bnb"

    def fields(self, fields, x, y):
        out = medium(to_torch(fields), torch.as_tensor(x), torch.as_tensor(y))
        return tuple(v.numpy() for v in out)

    def flight_tick(self, ship, power, stick, fields, flight=None):
        p = PRESETS["squad"]
        sigma, pi = flight_mod.actions_to_controls(torch.as_tensor(power), torch.as_tensor(stick))
        ship_t = to_torch({k: ship[k] for k in flight_mod.FLIGHT_KEYS})
        ship_type = torch.as_tensor(ship.get("ship_type", np.zeros_like(ship["x"], dtype=np.int32)))
        F = flight_of(type_flights(flight), ship_type)
        out = flight_mod.flight_tick(F, p.h, ship_t, sigma, pi, to_torch(fields))
        return to_numpy({k: out[k] for k in flight_mod.FLIGHT_KEYS})

    def bullet_tick(self, bullets, fields):
        p = PRESETS["squad"]
        out = bullet_motion(p.h, to_torch({k: bullets[k] for k in BULLET_MOTION_KEYS}),
                            to_torch(fields))
        return to_numpy(out)

    def reset(self, preset, uniforms):
        return to_numpy(match.reset(preset_of(preset), torch.as_tensor(uniforms)))

    def step(self, preset, state, actions, spread, uniforms):
        p = preset_of(preset)
        st, events = physics_step(p, with_memory(to_torch(state)), torch.as_tensor(actions),
                                  torch.as_tensor(spread), match.new_map(p, torch.as_tensor(uniforms)))
        return to_numpy(st), to_numpy(events), to_numpy(race_observation(p, st))

    def observe(self, preset, state):
        return to_numpy(race_observation(preset_of(preset), to_torch(state)))

    def chaser(self, preset, state):
        from race_chaser import chaser

        return chaser(preset_of(preset), to_torch(state)).numpy()
