"""Yongma and Jujak as vizia handle deltas scaled from the WWII archetype ratios
(archetypes_ww2.json): each mapped handle is vizia's times r^(+s/2) for Jujak and r^(-s/2)
for Yongma, r the turn-over-energy ratio of its real quantity, so vizia sits at the
geometric midpoint and the pair's split is the real split to the power s. Solves each card
and writes archetypes_solved.json (what each flies, its time-to-kill budget).

Run: PYTHONPATH=src python docs/rewrite/sections/gun-study/archetypes_solve.py
"""

import dataclasses
import json
import math
import pathlib
import tomllib

import torch

from bnb.game import api
from bnb.game.handling import Handles
from bnb.game.handling.characteristics import characteristics

HERE = pathlib.Path(__file__).parent
SHIP = HERE.parents[3] / "src/bnb/game/ships/real-scale/vizia.toml"

# handle: (real quantity, how the handle follows it, scaling group). "same" scales the handle
# with the ratio, "inverse" with its inverse.
MAPPING = {
    "stall_speed": ("stall_speed", "same", "turn"),
    "sustained_turn_rate": ("sustained_turn", "same", "turn"),
    "low_speed_acceleration": ("low_speed_acceleration", "same", "turn"),
    "climb_time": ("climb_rate", "inverse", "turn"),
    "cruise_speed": ("top_speed_altitude", "same", "speed"),
    "zero_lift_drag": ("ballistic_coefficient", "inverse", "speed"),
    "dive_acceleration": ("dive_limit", "same", "speed"),
    "max_shield": ("shield_rule", "same", "guns"),
    "hit_damage": ("weight_of_fire", "same", "guns"),
    "hit_radius": ("half_span", "same", "turn"),
}
# The set's survivability rule (real_scale.md): shield 100 x (1 + 0.2 per rating step).
SCALINGS = {
    "s1": dict(turn=1.0, speed=1.0, guns=0.5),
    "s1-speed2": dict(turn=1.0, speed=2.0, guns=0.5),
}


def ratios():
    data = json.load(open(HERE / "archetypes_ww2.json"))
    r = dict(data["turn_over_energy"])
    turn, energy = data["group_means"]["turn"], data["group_means"]["energy"]
    r["shield_rule"] = (1 + 0.2 * (turn["survivability"] - 2)) / (
        1 + 0.2 * (energy["survivability"] - 2)
    )
    return r


def proposals(scaling):
    r = ratios()
    base = tomllib.load(open(SHIP, "rb"))["handles"]
    out = {"vizia": {}, "Yongma": {}, "Jujak": {}}
    for handle, (quantity, how, group) in MAPPING.items():
        exponent = SCALINGS[scaling][group] / 2 * (1 if how == "same" else -1)
        factor = r[quantity] ** exponent
        out["Jujak"][handle] = base[handle] * factor
        out["Yongma"][handle] = base[handle] / factor
    return base, out


def solve(base, delta, game_rules):
    table = dict(base, **delta)
    handles = Handles(**{n: torch.tensor([v], dtype=torch.float64) for n, v in table.items()})
    solution = api.solve_handles(handles, game_rules, strict=False)
    t = solution.types
    t32 = type(t)(**{f.name: getattr(t, f.name).float() for f in dataclasses.fields(t)})
    c = characteristics(t32, game_rules)
    row = dict(
        converged=bool(solution.converged[0]),
        worst_error=float(solution.error.abs().amax()),
        cruise=float(c.cruise_speed[0]),
        top=float(c.top_speed[0]),
        corner=float(c.corner_speed[0]),
        turn_deg=[round(math.degrees(x), 2) for x in c.turn_rate[0].tolist()],
        instant_deg=math.degrees(float(c.max_acceleration[0] / c.corner_speed[0])),
        gun_range=float(c.gun_range[0]),
        climb_power=float(c.climb_power[0]),
        store=float(c.max_energy[0]),
        failures=list(solution.failures),
    )
    row["turn_180_sustained"] = 180 / row["turn_deg"][2]
    row["turn_180_instant"] = 180 / row["instant_deg"]
    row["turn_circle"] = 2 * row["cruise"] / math.radians(row["turn_deg"][2])
    return row, table


def time_to_kill(shooter, target, aspect_fraction):
    hits = math.ceil(target["max_shield"] / (shooter["hit_damage"] * aspect_fraction) - 1e-9)
    return (hits - 1) / shooter["fire_rate"]


def main():
    from bench.gun_lethality import setup

    _, _, game_rules = setup()
    out = {}
    for scaling in SCALINGS:
        base, deltas = proposals(scaling)
        flown, tables = {}, {}
        for name, delta in deltas.items():
            flown[name], tables[name] = solve(base, delta, game_rules)
        budget = {}
        for a in ("Yongma", "Jujak"):
            for b in ("Yongma", "Jujak"):
                pass_time = max(flown[a]["gun_range"], flown[b]["gun_range"]) / (
                    flown[a]["cruise"] + flown[b]["cruise"]
                )
                head_on = time_to_kill(tables[a], tables[b], tables[b]["head_on_fraction"])
                budget[f"{a} shoots {b}"] = dict(
                    head_on=head_on,
                    passes=head_on / pass_time,
                    tail=time_to_kill(tables[a], tables[b], 1.0),
                    tail_over_target_turn=time_to_kill(tables[a], tables[b], 1.0)
                    / flown[b]["turn_180_instant"],
                )
        out[scaling] = dict(scaling=SCALINGS[scaling], delta=deltas, flies=flown, budget=budget)
        print(scaling, json.dumps(dict(flies=flown, budget=budget), indent=1))
    json.dump(out, open(HERE / "archetypes_solved.json", "w"), indent=1)


if __name__ == "__main__":
    main()
