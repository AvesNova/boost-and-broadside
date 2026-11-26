# ruff: noqa: E501
"""Post-stall regime near stall speed: the nose, firing and a long hard turn, one ship, no
fields, the game's own step (lean.sandbox.world.World: core.tick) or flight tick.

1. The nose: full stick from straight flight at 1.0, 1.2 and 1.5 x the stall (1 g) speed: peak
   angle of attack, its time, and the peak angle rate (d angle / dt) in the first second.
2. Firing: 5 s of continuous fire from 1.2 x stall speed, coasting straight and at full stick:
   speed and store every second, and the shots fired.
3. Hard turning alone: 10 s of full stick from 1.05 x corner speed: the lowest speed and the store.

    PYTHONPATH=. uv run --no-sync python docs/rewrite/sections/post-stall/slow_checks.py
"""

import dataclasses
import math

import torch

from lean.game import core
from lean.sandbox.world import World

torch.set_num_threads(1)
FULL, STRAIGHT, COAST = 6, 0, 0


def world(speed: float) -> World:
    config = dataclasses.replace(core.GameConfig(clock_ticks=10**9), field_count=0)
    w = World(config, player_type=0, opponent_on=False)
    w.state.speed[0, 0] = speed
    return w


def constants():
    return core.per_type_constants(core.GameConfig(), torch.tensor([0]))


def nose():
    p = constants()
    stall = float(p.stall_speed[0])
    print("nose, full stick from straight: v/stall | peak AoA deg (t s) | peak AoA rate deg/s | time to 30 deg s | AoA at 1 s")
    for share in (1.0, 1.2, 1.5, 2.0):
        w = world(share * stall)
        angles = []
        for _ in range(60):
            w.advance(FULL, COAST, False)
            angles.append(float(w.state.angle[0, 0]))
        rates = [(b - a) * 30.0 for a, b in zip([0.0] + angles, angles)]
        peak = max(range(60), key=lambda i: angles[i])
        reach = next((i for i, a in enumerate(angles) if a >= math.radians(30.0)), None)
        to_30 = f"{(reach + 1) / 30:.2f}" if reach is not None else "never (2 s)"
        print(
            f"  {share:3.1f} | {math.degrees(angles[peak]):5.1f} ({(peak + 1) / 30:.2f}) | "
            f"{math.degrees(max(rates)):6.1f} | {to_30} | {math.degrees(angles[29]):5.1f}"
        )


def firing():
    p = constants()
    stall, top = float(p.stall_speed[0]), float(p.max_energy[0])
    print(f"5 s of fire from 1.2 x stall ({1.2 * stall:.1f} m/s; stall {stall:.1f}): speed m/s / store fraction each second, shots")
    for stick, name in ((STRAIGHT, "straight"), (FULL, "full stick")):
        w = world(1.2 * stall)
        marks, shots = [], 0
        for tick in range(150):
            w.advance(stick, COAST, True)
            shots += int(bool(w.state.fired[0, 0]))
            if (tick + 1) % 30 == 0:
                marks.append(f"{float(w.state.speed[0, 0]):5.1f}/{float(w.state.energy[0, 0]) / top:.3f}")
        print(f"  {name:10s} start {1.2 * stall:5.1f}/0.500 | " + " | ".join(marks) + f" | shots {shots}")


def long_turn():
    p = constants()
    corner, top = float(p.corner_speed[0]), float(p.max_energy[0])
    w = world(1.05 * corner)
    speeds = []
    for _ in range(300):
        w.advance(FULL, COAST, False)
        speeds.append(float(w.state.speed[0, 0]))
    print(
        f"10 s full stick from 1.05 x corner: lowest {min(speeds):.1f} m/s, end {speeds[-1]:.1f} m/s, "
        f"store {float(w.state.energy[0, 0]) / top:.3f} (stall {float(p.stall_speed[0]):.1f})"
    )


if __name__ == "__main__":
    nose()
    firing()
    long_turn()
