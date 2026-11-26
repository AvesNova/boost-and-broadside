# ruff: noqa: E501
"""Post-stall regime: a pull and a release flown with the game's flight tick (core.flight_tick),
one ship, no fields, coasting at the spawn store, with GameConfig.post_stall on and off.

For each start speed (shares of corner speed): the stick (full or middle) held PULL seconds
from straight flight, then released (straight) RELEASE seconds. Printed per flight: the peak
angle of attack and its overshoot over the commanded angle at that moment, the time to the
peak, the angle at the end of the pull; the path turn rate and nose rate at 0.25 s and at the
end; the peak lateral acceleration over the g limit; the speed lost over 0.5 s and 1 s; the
angle's overshoot past centre after the release (as a share of the angle at release).

Then the quasi-steady sweep through corner (core.steady_rates, each level's own shape:
core.stick_level_shape): the commanded angle and the path turn rate of middle and full stick,
fine steps from 0.80 to 1.20 x corner, and the largest step-to-step change (no jump).

    PYTHONPATH=. uv run --no-sync python docs/rewrite/sections/post-stall/flight_check.py
"""

import dataclasses
import math
from types import SimpleNamespace

import torch

from lean.game import core

torch.set_num_threads(1)
SHARES = (0.6, 0.8, 0.95, 1.0, 1.1)
PULL, RELEASE = 1.0, 1.0  # seconds
FULL, MIDDLE, STRAIGHT = 6, 4, 0  # stick indices (card.STICK_LEVELS: +3 Snap, +2 Max, 0)
COAST = 0


def configured(post_stall: bool) -> core.GameConfig:
    return dataclasses.replace(core.GameConfig(), post_stall=post_stall, field_count=0)


def fly(config, speed: float, stick: int):
    """Per tick: time, angle, commanded angle, path rate, nose rate, speed, lateral acceleration."""
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
    h = config.tick_length
    power = torch.zeros(1, 1, dtype=torch.long)
    size, sustain, deep = core.stick_level_shape(p, stick)
    rows = []
    with torch.no_grad():
        for tick in range(round((PULL + RELEASE) / h)):
            held = tick < round(PULL / h)
            index = stick if held else STRAIGHT
            path_before = math.atan2(float(s.heading_sin), float(s.heading_cos)) - float(s.angle)
            s = core.flight_tick(
                constants, torch.zeros(1, 5, 0), h, s, torch.full((1, 1), index), power
            )
            path_after = math.atan2(float(s.heading_sin), float(s.heading_cos)) - float(s.angle)
            turn = (path_after - path_before + math.pi) % (2 * math.pi) - math.pi
            v = float(s.speed)
            # the commanded angle at this speed (the steady analysis' own function)
            speed_t = torch.tensor([v])
            control = core.steady_control(
                p,
                speed_t,
                torch.tensor([size if held else 0.0]),
                COAST,
                torch.tensor([sustain if held else 0.0]),
                torch.tensor([deep if held else 0.0]),
            )
            lateral, _, vertical, _ = core._setting(
                p, control, speed_t, s.energy[0], s.angle_sin[0].abs()
            )
            commanded = float(
                core._commanded_angle(p, control, lateral, speed_t, 1.0 / p.lift, vertical)
            )
            rows.append(
                (
                    (tick + 1) * h,
                    float(s.angle),
                    commanded,
                    turn / h,
                    float(s.nose_rate),
                    v,
                    v * turn / h,
                )
            )
    return rows, float(p.max_acceleration[0])


def at(rows, time):
    return min(rows, key=lambda row: abs(row[0] - time))


def summary(rows, limit, start):
    pull = [row for row in rows if row[0] <= PULL + 1e-9]
    release = [row for row in rows if row[0] > PULL + 1e-9]
    peak = max(pull, key=lambda row: row[1])
    end = pull[-1]
    past_centre = min(row[1] for row in release)
    return dict(
        peak=math.degrees(peak[1]),
        peak_time=peak[0],
        overshoot=(peak[1] / peak[2] - 1.0) if peak[2] > 1e-6 else float("nan"),
        end_angle=math.degrees(end[1]),
        end_commanded=math.degrees(end[2]),
        path_quarter=math.degrees(at(pull, 0.25)[3]),
        nose_quarter=math.degrees(at(pull, 0.25)[4]),
        path_end=math.degrees(end[3]),
        nose_end=math.degrees(end[4]),
        peak_g=max(row[6] for row in pull) / limit,
        lost_half=start - at(pull, 0.5)[5],
        lost_one=start - end[5],
        release_overshoot=-past_centre / end[1] if end[1] > 1e-6 else float("nan"),
    )


def flights():
    print(
        "level  regime  v/corner  v0  | peak deg (t s) overshoot | end deg (cmd) | path/nose deg/s"
        " @0.25 s, @1 s | peak lateral/g limit | lost m/s 0.5 s, 1 s | release past centre"
    )
    for stick, name in ((FULL, "full"), (MIDDLE, "middle")):
        for post_stall in (True, False):
            config = configured(post_stall)
            corner = float(core.per_type_constants(config, torch.tensor([0])).corner_speed[0])
            for share in SHARES:
                start = share * corner
                rows, limit = fly(config, start, stick)
                r = summary(rows, limit, start)
                print(
                    f"{name:6s} {'on' if post_stall else 'off':3s}  {share:4.2f}  {start:5.1f} | "
                    f"{r['peak']:5.1f} ({r['peak_time']:.2f}) {100 * r['overshoot']:+5.1f}% | "
                    f"{r['end_angle']:5.1f} ({r['end_commanded']:5.1f}) | "
                    f"{r['path_quarter']:5.1f}/{r['nose_quarter']:5.1f}, "
                    f"{r['path_end']:5.1f}/{r['nose_end']:5.1f} | {r['peak_g']:5.3f} | "
                    f"{r['lost_half']:5.1f}, {r['lost_one']:5.1f} | "
                    f"{100 * r['release_overshoot']:4.1f}%"
                )


def sweep():
    config = configured(True)
    p = core.per_type_constants(config, torch.tensor([0]))
    p = SimpleNamespace(
        **{k: v.double() if v.is_floating_point() else v for k, v in vars(p).items()}
    )
    corner = float(p.corner_speed[0])
    speeds = torch.linspace(0.80 * corner, 1.20 * corner, 401, dtype=torch.float64)
    energy = p.spawn_energy.expand_as(speeds)
    out = {}
    for stick, name in ((FULL, "full"), (MIDDLE, "middle")):
        size, sustain, deep = core.stick_level_shape(p, stick)
        rate = core.steady_rates(
            p, speeds, torch.full_like(speeds, size), COAST, energy, sustain, deep
        )[0]
        control = core.steady_control(
            p, speeds, torch.full_like(speeds, size), COAST, sustain, deep
        )
        lateral, _, vertical, _ = core._setting(p, control, speeds, energy, 0.0)
        angle = core._commanded_angle(p, control, lateral, speeds, 1.0 / p.lift, vertical)
        out[name] = (rate, angle)
        print(
            f"{name}: largest step (0.001 x corner) in rate {math.degrees(float(rate.diff().abs().max())):.3f}"
            f" deg/s, in commanded angle {math.degrees(float(angle.diff().abs().max())):.3f} deg"
        )
    near = (speeds > 0.98 * corner) & (speeds < 1.02 * corner)
    for name, (rate, angle) in out.items():
        print(
            f"{name} within 2% of corner: largest step in rate"
            f" {math.degrees(float(rate[near].diff().abs().max())):.3f} deg/s, in angle"
            f" {math.degrees(float(angle[near].diff().abs().max())):.4f} deg"
        )
    top = core.steady_rates(p, speeds, torch.ones_like(speeds), COAST, energy)[0]
    print(
        "v/corner  full: angle deg, rate deg/s | middle: angle, rate | top line rate | lift/g limit (full)"
    )
    for i in range(0, 401, 20):
        v = speeds[i]
        full_rate, full_angle = out["full"][0][i], out["full"][1][i]
        lateral_lift = core._lift_coefficient(p, full_angle) * v * v
        print(
            f"{float(v) / corner:5.3f}  {math.degrees(float(full_angle)):5.1f} {math.degrees(float(full_rate)):5.1f} | "
            f"{math.degrees(float(out['middle'][1][i])):5.1f} {math.degrees(float(out['middle'][0][i])):5.1f} | "
            f"{math.degrees(float(top[i])):5.1f} | {float(lateral_lift / p.max_acceleration[0]):.3f}"
        )
    better = out["middle"][0] > out["full"][0]
    below = speeds < corner
    crossing = (
        float(speeds[below & ~better].min()) / corner if bool((below & ~better).any()) else None
    )
    print(f"middle turns faster than full below corner from 0.80 x corner up to {crossing}")


def lowest_speeds():
    """Below 0.8 x corner the steady rates of full, middle and the top line, down to stall."""
    config = configured(True)
    p = core.per_type_constants(config, torch.tensor([0]))
    p = SimpleNamespace(
        **{k: v.double() if v.is_floating_point() else v for k, v in vars(p).items()}
    )
    corner = float(p.corner_speed[0])
    speeds = torch.tensor([0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=torch.float64) * corner
    energy = p.spawn_energy.expand_as(speeds)
    print(
        "v/corner  full rate, d speed/dt | middle rate, d speed/dt | top line rate (deg/s, m/s^2)"
    )
    rows = {}
    for stick in (FULL, MIDDLE):
        size, sustain, deep = core.stick_level_shape(p, stick)
        rows[stick] = core.steady_rates(
            p, speeds, torch.full_like(speeds, size), COAST, energy, sustain, deep
        )
    top = core.steady_rates(p, speeds, torch.ones_like(speeds), COAST, energy)[0]
    for i, v in enumerate(speeds):
        print(
            f"{float(v) / corner:4.2f}  {math.degrees(float(rows[FULL][0][i])):5.1f} {float(rows[FULL][1][i]):7.1f} | "
            f"{math.degrees(float(rows[MIDDLE][0][i])):5.1f} {float(rows[MIDDLE][1][i]):7.1f} | {math.degrees(float(top[i])):5.1f}"
        )


def drag_ratio():
    """The drag coefficient at 45 deg over the one at the stall (straight, no held vertical)."""
    p = core.per_type_constants(configured(True), torch.tensor([0]))
    speed = 0.5 * p.corner_speed
    setting = (None, torch.zeros(1), torch.zeros(1), torch.zeros(1))

    def coefficient(angle):
        angle = torch.tensor([angle])
        lift = core._lift_coefficient(p, angle)
        drag = core._rates(
            p, setting, None, speed, lift, torch.cos(angle), 1.0 / speed**2, 1.0 / p.lift**2
        )[0]
        return -float(drag) / float(speed) ** 2

    stall = float(p.stall[0])
    for degrees in (30.0, 45.0, 60.0, 90.0):
        print(
            f"drag at {degrees:.0f} deg over drag at the stall: "
            f"{coefficient(math.radians(degrees)) / coefficient(stall):.2f}"
        )
    for degrees in (15.5, 20.0, 25.0, 30.0, 35.0, 45.0, 60.0, 90.0):
        lift = core._lift_coefficient(p, torch.tensor([math.radians(degrees)]))
        print(f"lift at {degrees:4.1f} deg over the maximum: {float(lift / p.lift):.3f}")


if __name__ == "__main__":
    drag_ratio()
    print()
    flights()
    print()
    sweep()
    print()
    lowest_speeds()
