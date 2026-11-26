# ruff: noqa: E501
"""Post-stall regime, the step-8 measurements (Sustain, Max, Snap), all from the game's own
functions (the step-7 version of this file is in git history):

1. The quasi-steady lines (core.steady_rates, each level's own shape: core.stick_level_shape),
   coasting at the spawn store, a 0.1 m/s grid from 85 m/s to 470 m/s: per line its peak, the
   largest rise toward the stall below the peak (one grid step, and in all: the most any lower
   speed turns above a higher one), the steepest slope; Snap diving and climbing; the envelope
   (the most turn of any level). Sustain against a fresh bisection of the Ps = 0 stick.
2. The nose going out from straight at 0.8 x corner and 1.5 x stall speed (core.flight_tick, no
   fields): per level its damping ratio (nose_damping / sqrt(card.NOSE_STIFFNESS)), the peak
   angle over the commanded angle at that moment, the time to 90% of the commanded angle.
3. Reversals: Snap left held 1.5 s from 0.8 x corner, then straight / Sustain right / Max right /
   Snap right: the time for the nose to cross centre and the overshoot on the far side.
4. Held (lean.sandbox.world.World, core.tick, 60 s from the spawn): Sustain, Max, Snap.

    PYTHONPATH=. uv run --no-sync python docs/rewrite/sections/post-stall/lines_check.py
"""

import dataclasses
import math
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(__file__))
from slow_checks import world  # noqa: E402

from lean.game import card, core  # noqa: E402

torch.set_num_threads(1)
STRAIGHT, COAST = 0, 0
LEFT = {"sustain": 2, "max": 4, "snap": 6}
RIGHT = {"sustain": 1, "max": 3, "snap": 5}
DEGREES = 180.0 / math.pi


def constants64():
    p = core.per_type_constants(core.GameConfig(), torch.tensor([0]))
    return SimpleNamespace(
        **{k: v.double() if v.is_floating_point() else v for k, v in vars(p).items()}
    )


def line(p, speeds, index, power=COAST):
    size, sustain, deep = core.stick_level_shape(p, index)
    energy = p.spawn_energy.expand_as(speeds)
    return core.steady_rates(p, speeds, torch.full_like(speeds, size), power, energy, sustain, deep)


def lines():
    p = constants64()
    speeds = torch.arange(850, 4701, dtype=torch.float64) / 10.0
    rates = {name: line(p, speeds, index)[0] for name, index in LEFT.items()}
    rates["snap dive"] = line(p, speeds, LEFT["snap"], 1)[0]
    rates["snap climb"] = line(p, speeds, LEFT["snap"], 2)[0]
    rates["envelope"] = torch.maximum(torch.maximum(rates["max"], rates["snap"]), rates["sustain"])
    print(
        f"lines, 0.1 m/s grid {float(speeds[0]):.1f}-{float(speeds[-1]):.1f} m/s (stall {float(p.stall_speed[0]):.1f}, corner {float(p.corner_speed[0]):.1f}, cruise {float(p.cruise_speed[0]):.1f})"
    )
    for name, rate in rates.items():
        degrees = rate * DEGREES
        peak = int(degrees.argmax())
        below = degrees[: peak + 1]
        rise = below[:-1] - below[1:]
        i = int(rise.argmax())
        best_lower = torch.cummax(below, 0)  # the most turn at this speed or any lower one
        total = best_lower.values - below
        j = int(total.argmax())
        k = int(best_lower.indices[j])
        slope = degrees.diff() / 0.1
        m = int(slope.abs().argmax())
        print(
            f"  {name:10s} peak {float(degrees[peak]):5.2f} at {float(speeds[peak]):5.1f} | rise toward the stall: step {float(rise[i]):+.4f} at {float(speeds[i]):.1f}, "
            f"in all {float(total[j]):.2f} deg/s ({float(below[k]):.2f} at {float(speeds[k]):.1f} over {float(below[j]):.2f} at {float(speeds[j]):.1f}) | "
            f"steepest {float(slope[m]):+.3f} deg/s per m/s at {float(speeds[m]):.1f}"
        )
    print("  v m/s | Sustain | Max | Snap | envelope (deg/s)")
    for v in (
        85.0,
        88.0,
        90.0,
        92.0,
        95.0,
        98.0,
        100.0,
        104.0,
        110.0,
        120.0,
        147.0,
        167.0,
        200.0,
        255.0,
        300.0,
        350.0,
        370.0,
    ):
        i = int(torch.argmin((speeds - v).abs()))
        print(
            "  "
            + f"{v:5.1f} | "
            + " | ".join(
                f"{float(rates[n][i]) * DEGREES:6.2f}"
                for n in ("sustain", "max", "snap", "envelope")
            )
        )
    # Sustain's table against a fresh bisection of the stick holding Ps = 0 (Max's sweep).
    check = torch.linspace(150.0, 365.0, 44, dtype=torch.float64)
    energy = p.spawn_energy.expand_as(check)
    low, high = torch.zeros_like(check), torch.ones_like(check)
    for _ in range(40):
        middle = 0.5 * (low + high)
        gains = core.quasi_steady(p, check, middle, COAST, energy)[2] > 0.0
        low, high = torch.where(gains, middle, low), torch.where(gains, high, middle)
    exact = core.steady_rates(p, check, 0.5 * (low + high), COAST, energy)[0]
    table = line(p, check, LEFT["sustain"])[0]
    gap = (table - exact) * DEGREES
    i = int(gap.abs().argmax())
    print(
        f"  Sustain (table) minus the bisected Ps = 0 turn, 150-365 m/s: largest {float(gap[i]):+.3f} deg/s at {float(check[i]):.1f} m/s"
    )


def ship(speed: float):
    config = dataclasses.replace(core.GameConfig(), field_count=0)
    constants = core.ship_constants(config, torch.zeros(1, 1, dtype=torch.long))
    p = core.per_type_constants(config, torch.tensor([0]))
    z = torch.zeros(1, 1)
    s = SimpleNamespace(
        x=z.clone(),
        y=z.clone(),
        heading_cos=torch.ones(1, 1),
        heading_sin=z.clone(),
        speed=torch.full((1, 1), speed),
        angle=z.clone(),
        angle_cos=torch.ones(1, 1),
        angle_sin=z.clone(),
        nose_rate=z.clone(),
        energy=torch.full((1, 1), float(p.spawn_energy[0])),
        log_index=z.clone(),
        gradient_x=z.clone(),
        gradient_y=z.clone(),
    )
    return config, constants, p, s


def commanded(p, index, s):
    """The angle the tick steers toward at this state (core._commanded_angle at the ship's speed
    and store, the 1 g hold at its angle)."""
    size, sustain, deep = core.stick_level_shape(p, index)
    speed = s.speed[0]
    control = core.steady_control(
        p, speed, torch.tensor([size]), COAST, torch.tensor([sustain]), torch.tensor([deep])
    )
    control.stick_sign = torch.sign(p.stick_table[:, index])
    lateral, _, vertical, _ = core._setting(p, control, speed, s.energy[0], s.angle_sin[0].abs())
    return float(core._commanded_angle(p, control, lateral, speed, 1.0 / p.lift, vertical))


def fly(config, constants, p, s, index, seconds):
    rows = []
    with torch.no_grad():
        for tick in range(round(seconds / config.tick_length)):
            s = core.flight_tick(
                constants,
                torch.zeros(1, 5, 0),
                config.tick_length,
                s,
                torch.full((1, 1), index),
                torch.zeros(1, 1, dtype=torch.long),
            )
            rows.append(
                (
                    (tick + 1) * config.tick_length,
                    float(s.angle),
                    commanded(p, index, s),
                    float(s.speed),
                )
            )
    return s, rows


def going_out():
    print(
        "going out from straight (flight tick, 1.5 s): level | damping ratio | peak AoA deg (t s) over commanded then | time to 90% of commanded s"
    )
    for label, share in (
        ("0.8 x corner", ("corner_speed", 0.8)),
        ("1.5 x stall", ("stall_speed", 1.5)),
    ):
        for name, index in LEFT.items():
            config, constants, p, s = ship(0.0)
            s.speed.fill_(share[1] * float(getattr(p, share[0])[0]))
            ratio = float(p.nose_damping[0]) / math.sqrt(
                card.NOSE_STIFFNESS[int(abs(card.STICK_LEVELS[index]))]
            )
            _, rows = fly(config, constants, p, s, index, 1.5)
            peak = max(rows, key=lambda r: r[1])
            reach = next((r for r in rows if r[2] > 1e-6 and r[1] >= 0.9 * r[2]), None)
            print(
                f"  {label:12s} {name:8s} | {ratio:.3f} | {peak[1] * DEGREES:5.1f} ({peak[0]:.2f}) over {peak[2] * DEGREES:5.1f}: {100 * (peak[1] / peak[2] - 1):+5.1f}% | "
                + (f"{reach[0]:.2f} (commanded {reach[2] * DEGREES:.1f})" if reach else "never")
            )


def reversals():
    print(
        "reversals: Snap left 1.5 s from 0.8 x corner, then: AoA and speed at the switch | time to cross centre s | far side: peak AoA over commanded then"
    )
    for name, index in (
        ("release", STRAIGHT),
        ("Sustain right", RIGHT["sustain"]),
        ("Max right", RIGHT["max"]),
        ("Snap right", RIGHT["snap"]),
    ):
        config, constants, p, s = ship(0.0)
        s.speed.fill_(0.8 * float(p.corner_speed[0]))
        s, _ = fly(config, constants, p, s, LEFT["snap"], 1.5)
        at, speed = float(s.angle), float(s.speed)
        s, rows = fly(config, constants, p, s, index, 2.0)
        cross = next((r[0] for r in rows if r[1] <= 0.0), None)
        far = min(rows, key=lambda r: r[1])
        if index == STRAIGHT:
            far_text = f"{-far[1] * DEGREES:.2f} deg past centre ({100 * -far[1] / at:.1f}% of the angle at the switch, t {far[0]:.2f})"
        else:
            far_text = f"{-far[1] * DEGREES:.1f} deg (t {far[0]:.2f}) over {-far[2] * DEGREES:.1f}: {100 * (far[1] / far[2] - 1):+.1f}%"
        cross_text = f"{cross:.2f}" if cross is not None else "never"
        print(
            f"  {name:13s} | {at * DEGREES:5.1f} deg, {speed:5.1f} m/s | {cross_text} | {far_text}"
        )


def held():
    print(
        "held 60 s from the spawn (World): speed m/s, turn deg/s, AoA deg, store (speed/store at 10, 30, 60 s)"
    )
    for name, stick in LEFT.items():
        w = world(None)
        flown = []
        for _ in range(60 * 30):
            w.advance(stick, COAST, False)
            flown.append((w.ground_speed(), w.store_fraction()))
        marks = [f"{flown[t - 1][0]:.1f}/{flown[t - 1][1]:.4f}" for t in (300, 900, 1800)]
        r = w.readout()
        print(
            f"  {name:8s} {r['speed']:6.1f} {r['turn_rate_degrees']:6.2f} {r['angle_of_attack_degrees']:5.1f} {r['store']:.4f} ({', '.join(marks)})"
        )


if __name__ == "__main__":
    lines()
    print()
    going_out()
    print()
    reversals()
    print()
    held()
