"""Feasibility of the two proposed vizia variants: solve each card, report what it flies."""

import dataclasses
import json
import math
import tomllib

import torch

from bench.gun_lethality import setup
from bnb.game import api
from bnb.game.handling import Handles
from bnb.game.handling.characteristics import characteristics

game, types, rules = setup()
base = tomllib.load(open("src/bnb/game/ships/real-scale/vizia.toml", "rb"))["handles"]
D = math.radians
PROPOSALS = {
    "vizia": {},
    "Yongma": dict(
        stall_speed=50.0,
        cruise_speed=182.0,
        sustained_turn_rate=D(13.5),
        ceiling=base["ceiling"] * 1.2,
        spawn_energy=base["spawn_energy"] * 1.2,
        dive_acceleration=base["dive_acceleration"] * 1.2,
        zero_lift_drag=base["zero_lift_drag"] * 0.9,
        half_lift_drag=0.20,
        max_shield=120.0,
        hit_radius=11.0,
    ),
    "Jujak": dict(
        stall_speed=40.0,
        cruise_speed=150.0,
        sustained_turn_rate=D(18.5),
        ceiling=base["ceiling"] * 0.9,
        spawn_energy=base["spawn_energy"] * 0.9,
        dive_acceleration=base["dive_acceleration"] * 0.85,
        low_speed_acceleration=base["low_speed_acceleration"] * 1.1,
        nose_reversal_time=0.70,
        zero_lift_drag=base["zero_lift_drag"] * 1.1,
        half_lift_drag=0.30,
        max_shield=90.0,
        hit_radius=9.0,
    ),
}
out = {}
for name, delta in PROPOSALS.items():
    table = dict(base, **delta)
    handles = Handles(**{n: torch.tensor([v], dtype=torch.float64) for n, v in table.items()})
    solution = api.solve_handles(handles, rules, strict=False)
    t = solution.types
    t32 = type(t)(**{f.name: getattr(t, f.name).float() for f in dataclasses.fields(t)})
    c = characteristics(t32, rules)
    row = dict(
        converged=bool(solution.converged[0]),
        worst_error=float(solution.error.abs().amax()),
        cruise=float(c.cruise_speed[0]),
        top=float(c.top_speed[0]),
        corner=float(c.corner_speed[0]),
        turn_deg=[round(math.degrees(x), 1) for x in c.turn_rate[0].tolist()],
        instant_deg=math.degrees(float(c.max_acceleration[0] / c.corner_speed[0])),
        gun_range=float(c.gun_range[0]),
        climb_power=float(c.climb_power[0]),
        store=float(c.max_energy[0]),
        failures=list(solution.failures),
    )
    row["turn_180_sustained"] = 180 / row["turn_deg"][2]
    row["turn_180_instant"] = 180 / row["instant_deg"]
    row["turn_circle"] = 2 * row["cruise"] / math.radians(row["turn_deg"][2])
    out[name] = dict(delta={k: v for k, v in delta.items()}, flies=row)
    print(name, json.dumps(row))
json.dump(out, open("docs/rewrite/sections/gun-study/variants_solved.json", "w"), indent=1)
