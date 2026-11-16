"""The time a full-stick turn takes to reach its full rate from straight flight, per ship of
the real-scale set and speed (in corner speeds): coasting with a full store, the flight
path's rate per tick, and the first time it reaches 90% of its peak over 3 s. Fits the
lead the initiative turn takes (`imugi_combat.TURN_ONSET_IN_NOSE_PERIODS`).

    PYTHONPATH=src:. python docs/rewrite/sections/imugi-initiative/turn_onset.py
"""

import dataclasses

import torch

from bench import game_loop as gl
from bnb.game import api
from bnb.game.handling.shown import trajectory
from bnb.game.physics.flight import airframe_of

types = gl.base_card("cpu")
game = dataclasses.replace(api.PRESETS[gl.MODES["deathmatch"]], ships_per_team=1)
rules = api.rules_of(game, types=types)
airframe = airframe_of(types, torch.arange(2))
fractions = torch.tensor([0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.8])
ticks = round(3.0 / rules.tick_length)
batched = type(types)(**{f.name: getattr(types, f.name)[None] for f in dataclasses.fields(types)})
for k, name in enumerate(("Yongma", "Jujak")):
    corner = float(airframe.corner_speed[k])
    speed = (fractions * corner)[None, None].expand(1, 2, -1).clone()
    energy = types.max_energy[None, :, None].expand(1, 2, len(fractions)).clone()
    zeros = torch.zeros_like(speed)
    flown = trajectory(rules, batched, speed, energy, zeros, zeros, torch.ones_like(speed), zeros, ticks)
    path = flown["path"][:, 0, k]  # (ticks + 1, grid)
    rate = (path[1:] - path[:-1]) / rules.tick_length
    peak = rate.amax(0)
    reached = (rate >= 0.9 * peak).float().argmax(0) + 1
    omega = float(types.corner_nose_frequency[k])
    print(f"{name}: corner {corner:.0f} m/s, nose frequency at corner {omega:.2f} rad/s")
    for i, f in enumerate(fractions.tolist()):
        t90 = float(reached[i]) * rules.tick_length
        frequency = max(omega * f, rules.flight.nose_frequency_floor)
        print(
            f"  v/vc {f:.2f}: peak rate {float(peak[i]):.3f} rad/s, t90 {t90:.3f} s, "
            f"t90 x nose frequency {t90 * frequency:.2f}, turned by t90 {float(path[int(reached[i]), i]):.3f} rad"
        )
