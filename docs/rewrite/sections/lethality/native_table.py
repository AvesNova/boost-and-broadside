"""Game v13's native table for Yongma, Jujak and Vizia: stall, corner, cruise and top boost
speed, the settled turns (fine / sustained / max stick) and the time to kill, native and
plane-equivalent (native / real_scale.EQUIVALENT_FACTOR through the similarity).

    uv run --no-sync python docs/rewrite/sections/lethality/native_table.py
"""

import math

import torch

from bnb.contracts import GameConfig
from bnb.game import api, ships
from bnb.game.handling.shown import shown, sticks
from bnb.game.real_scale import EQUIVALENT_FACTOR as K

SET, KEYS = "real-scale", ("yongma", "jujak", "vizia")
rules = api.rules_of(GameConfig(1, 1, 7800.0, ship_set=SET))
order = [s.key for s in ships.load(SET)]
full = api.ship_types(ship_set=SET, dtype=torch.float64)
index = torch.tensor([order.index(k) for k in KEYS])
types = type(full)(**{n: v[index] for n, v in vars(full).items()})
view = shown(rules, types)
files = {k: ships.read(ships.directory_of(SET) / f"{k}.toml", k) for k in KEYS}
deg = 180.0 / math.pi
print("| | native | equivalent |\n|---|---|---|")
for i, key in enumerate(KEYS):
    h = files[key].handles
    speeds = dict(
        stall=h["stall_speed"],
        corner=float(view.corner_speed[0, i]),
        cruise=h["cruise_speed"],
        top_boost=float(view.top_speed[0, i]),
    )
    d = view.datum
    turns = [
        (float(d.speed[0, i, j]), float(d.turn_rate[0, i, j]) * deg) for j in range(3)
    ]
    shields = {k: files[k].card["max_shield"] for k in KEYS}
    rate = types.fire_rate[i].item()
    kills = [math.ceil(s / types.hit_damage[i].item()) + 1 for s in shields.values()]
    ttk = [(n - 1) / rate for n in kills]  # the first hit at t = 0
    name = files[key].name
    for label, v in speeds.items():
        print(f"| {name} {label} speed | {v:.0f} m/s | {v / K:.0f} m/s |")
    for level, (u, w) in zip(("fine", "sustained", "max"), turns):
        print(f"| {name} settled {level} | {u:.0f} m/s at {w:.1f} deg/s | {u / K:.0f} m/s at {w / K:.1f} deg/s |")
    vs = ", ".join(f"{k} {n} hits {t:.2f} s" for k, n, t in zip(shields, kills, ttk))
    vs_eq = ", ".join(f"{k} {t * K:.2f} s" for k, t in zip(shields, ttk))
    print(f"| {name} core time to kill | {vs} | {vs_eq} |")
print("sticks", sticks(types).tolist())
