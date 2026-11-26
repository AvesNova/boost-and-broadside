# ruff: noqa: E501
"""Post-stall regime, the step-7 measurements, all from the game's own functions:

1. The quasi-steady lines (core.steady_rates, each level's own shape: core.stick_level_shape;
   the envelope: stick 1 of the top line), coasting at the spawn store, a 0.1 m/s grid from
   85 m/s to 470 m/s: per line the largest rise toward the stall (the rate at a lower speed
   above the next higher one), the steepest slope, and the envelope minus max(middle, full).
   Full stick's commanded angle (core._commanded_angle) and its normal force over 1 g at a few
   speeds.
2. Held (lean.sandbox.world.World, core.tick, 60 s from the spawn): full, middle, gradual.
3. Snaps: full stick from straight at 0.6 / 0.8 / 0.95 x corner (flight_check.fly).
4. Nose: full stick from straight at 1.0 / 1.2 / 1.5 x stall speed (World, 2 s): the commanded
   angle at the start speed and at the peak, and the peak angle.

    PYTHONPATH=. uv run --no-sync python docs/rewrite/sections/post-stall/lines_check.py
"""

import math
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(__file__))
from flight_check import FULL, configured, fly  # noqa: E402
from slow_checks import world  # noqa: E402

from lean.game import card, core  # noqa: E402

torch.set_num_threads(1)
MIDDLE, GRADUAL, COAST = 4, 2, 0
DEGREES = 180.0 / math.pi


def constants64():
    p = core.per_type_constants(core.GameConfig(), torch.tensor([0]))
    return SimpleNamespace(
        **{k: v.double() if v.is_floating_point() else v for k, v in vars(p).items()}
    )


def control_for(p, speeds, stick):
    size, cap, deep = core.stick_level_shape(p, stick)
    return SimpleNamespace(
        thrust_scale=1.0,
        stick_size=torch.full_like(speeds, size),
        stick_sign=torch.ones_like(speeds),
        stick_cap=torch.full_like(speeds, cap),
        deep=torch.full_like(speeds, deep),
        commanded=p.commanded_table[..., COAST],
        demand=p.demand_table[..., COAST],
    )


def commanded(p, speeds, stick, energy):
    control = control_for(p, speeds, stick)
    setting = core._setting(p, control, speeds, energy)
    return core._commanded_angle(p, control, setting[0], speeds, 1.0 / p.lift, setting[2]), setting


def lines():
    p = constants64()
    speeds = torch.arange(850, 4701, dtype=torch.float64) / 10.0
    energy = p.spawn_energy.expand_as(speeds)
    rates = {}
    for name, stick in (("gradual", GRADUAL), ("middle", MIDDLE), ("full", FULL)):
        size, cap, deep = core.stick_level_shape(p, stick)
        rates[name] = core.steady_rates(
            p, speeds, torch.full_like(speeds, size), COAST, energy, cap, deep
        )[0]
    rates["envelope"] = core.steady_rates(p, speeds, torch.ones_like(speeds), COAST, energy)[0]
    size, cap, deep = core.stick_level_shape(p, FULL)
    for power, name in ((1, "full dive"), (2, "full climb")):
        rates[name] = core.steady_rates(
            p, speeds, torch.full_like(speeds, size), power, energy, cap, deep
        )[0]
    print(
        f"lines, 0.1 m/s grid {float(speeds[0]):.1f}-{float(speeds[-1]):.1f} m/s (stall {float(p.stall_speed[0]):.1f}, corner {float(p.corner_speed[0]):.1f})"
    )
    for name, rate in rates.items():
        degrees = rate * DEGREES
        peak = int(degrees.argmax())  # toward the stall: below the line's peak
        rise = degrees[:peak] - degrees[1 : peak + 1]  # lower speed minus the next higher one
        i = int(rise.argmax())
        slope = degrees.diff() / 0.1
        j = int(slope.abs().argmax())
        print(
            f"  {name:10s} peak {float(degrees[peak]):5.2f} deg/s at {float(speeds[peak]):5.1f} m/s | largest rise toward the stall {float(rise[i]):+.4f} deg/s at {float(speeds[i]):.1f} m/s | "
            f"steepest slope {float(slope[j]):+.3f} deg/s per m/s at {float(speeds[j]):.1f} m/s"
        )
    best = torch.maximum(rates["middle"], rates["full"])
    gap = (rates["envelope"] - best) * DEGREES
    k, m = int(gap.argmax()), int(gap.argmin())
    print(
        f"  envelope - max(middle, full): largest {float(gap[k]):+.4f} deg/s at {float(speeds[k]):.1f} m/s, "
        f"smallest {float(gap[m]):+.4f} at {float(speeds[m]):.1f}"
    )
    angle, setting = commanded(p, speeds, FULL, energy)
    thrust = setting[3]
    lift = core._lift_coefficient(p, angle)
    normal = (lift * speeds * speeds + thrust * torch.sin(angle)) / card.GRAVITY
    wing = lift * speeds * speeds / card.GRAVITY
    # The force budget before `_path_lift`'s envelope cap (this check's own arithmetic, for the
    # report only): normal^2 less the share of the held vertical.
    share = core.smootherstep(((angle - p.stall) * p.inverse_share_width).clamp(0.0, 1.0))
    budget = torch.sqrt((normal * normal - share * (setting[2] / card.GRAVITY) ** 2).clamp_min(0.0))
    uncapped = budget * card.GRAVITY / speeds * DEGREES
    capped = (uncapped - rates["full"] * DEGREES) > 1e-3
    band = (speeds >= p.stall_speed) & (
        speeds <= p.stall_speed / math.sqrt(card.POST_STALL_LIFT_FRACTION)
    )
    print(
        f"  stall speed to stall speed / sqrt(lift fraction): wing lift lowest {float(wing[band].min()):.3f} g at"
        f" {float(speeds[band][wing[band].argmin()]):.1f} m/s, normal force lowest {float(normal[band].min()):.3f} g"
    )
    print(
        f"  the envelope cap binds on full stick at {int(capped.sum())} grid speeds, {float(speeds[capped].min()):.1f}-{float(speeds[capped].max()):.1f} m/s"
        f" (largest cut {float((uncapped - rates['full'] * DEGREES).max()):.2f} deg/s at {float(speeds[(uncapped - rates['full'] * DEGREES).argmax()]):.1f})"
        if bool(capped.any())
        else "  the envelope cap never binds on full stick"
    )
    print(
        "  v m/s | full: commanded deg, wing lift / 1 g, normal force (lift + thrust sin) / 1 g, turn deg/s (before the envelope cap) | envelope deg/s | middle deg/s"
    )
    for v in (
        85.0,
        90.0,
        92.0,
        95.0,
        98.0,
        100.0,
        102.0,
        104.0,
        106.0,
        110.0,
        120.0,
        140.0,
        160.0,
        200.0,
        220.0,
        250.0,
    ):
        i = int(torch.argmin((speeds - v).abs()))
        print(
            f"  {v:5.1f} | {float(angle[i]) * DEGREES:5.1f} {float(wing[i]):5.3f} {float(normal[i]):5.3f} {float(rates['full'][i]) * DEGREES:6.2f} ({float(uncapped[i]):6.2f}) | "
            f"{float(rates['envelope'][i]) * DEGREES:6.2f} | {float(rates['middle'][i]) * DEGREES:6.2f}"
        )


def held():
    print("held 60 s from the spawn (World): speed m/s, turn deg/s, AoA deg, store")
    for name, stick in (("full", FULL), ("middle", MIDDLE), ("gradual", GRADUAL)):
        w = world(None)
        flown = []
        for _ in range(60 * 30):
            w.advance(stick, COAST, False)
            flown.append((w.ground_speed(), w.store_fraction()))
        marks = [f"{flown[t - 1][0]:.1f}/{flown[t - 1][1]:.3f}" for t in (300, 900, 1800)]
        r = w.readout()
        lowest = min(speed for speed, _ in flown)
        print(
            f"  {name:8s} {r['speed']:6.1f} {r['turn_rate_degrees']:6.2f} {r['angle_of_attack_degrees']:5.1f} {r['store']:.3f} (lowest {lowest:.1f}; speed/store at 10, 30, 60 s: {', '.join(marks)})"
        )


def snaps():
    config = configured(True)
    corner = float(core.per_type_constants(config, torch.tensor([0])).corner_speed[0])
    print("snaps, full stick from straight: v/corner v0 | peak AoA deg (t s) | lost in 0.5 s m/s")
    for share in (0.6, 0.8, 0.95):
        start = share * corner
        rows, _ = fly(config, start, FULL)
        pull = [row for row in rows if row[0] <= 1.0 + 1e-9]
        peak = max(pull, key=lambda row: row[1])
        half = min(pull, key=lambda row: abs(row[0] - 0.5))
        print(
            f"  {share:4.2f} {start:5.1f} | {peak[1] * DEGREES:5.1f} ({peak[0]:.2f}) | {start - half[5]:5.1f}"
        )


def nose():
    p = constants64()
    stall = float(p.stall_speed[0])
    print(
        "nose, full stick from straight (World, 2 s): v/stall v0 | commanded deg at v0 | peak AoA deg (t s, v) | commanded at the peak"
    )
    for share in (1.0, 1.2, 1.5):
        w = world(share * stall)
        angles, speeds = [], []
        for _ in range(60):
            w.advance(FULL, COAST, False)
            angles.append(float(w.state.angle[0, 0]))
            speeds.append(float(w.state.speed[0, 0]))
        peak = max(range(60), key=lambda i: angles[i])
        at = torch.tensor([share * stall, speeds[peak]], dtype=torch.float64)
        angle, _ = commanded(p, at, FULL, p.spawn_energy.expand_as(at))
        print(
            f"  {share:3.1f} {share * stall:5.1f} | {float(angle[0]) * DEGREES:5.1f} | "
            f"{angles[peak] * DEGREES:5.1f} ({(peak + 1) / 30:.2f}, {speeds[peak]:.1f}) | {float(angle[1]) * DEGREES:5.1f}"
        )


if __name__ == "__main__":
    lines()
    print()
    held()
    print()
    snaps()
    print()
    nose()
