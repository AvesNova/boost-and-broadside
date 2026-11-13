"""Hard bounds of the one-tick ship-state changes from the game's own flight step
(docs/rewrite/sections/state-deltas.md, section 1).

1. The flight envelope, flown with the game's `flight_tick`: the terminal proper speed (full
   boost, straight, the store held full, so an upper bound) and the largest nose rate a
   full-stick reversal reaches, per ship type.
2. A sweep of `flight_tick` over a box around that envelope: proper speed 0 to the
   terminal speed, and the nose rate and angle of attack within 1.25x the largest the two
   players reached (OBSERVED: they are the nose oscillator's internal states, with no simple
   closed bound; refraction drives the nose rate, (u / n) |grad ln n|), any store, every
   stick and power, and positions over two overlapping fields of every pair of levels (ln n
   from -ln 2 to ln 2, gradients up to the overlap's). The largest one-tick change of each
   quantity over the box is its bound (the box holds states no ship reaches, such as the
   largest nose rate at the lowest speed, so the bound is conservative).

The discrete quantities (shield, recharge wait, gun cooldown) are bounded from the rules
directly. Writes <scratch>/bounds.json and prints a summary.
"""

import dataclasses
import json
import math
import pathlib

import torch

from bnb.game import api
from bnb.game.physics.flight import airframe_of, controls, flight_tick
from bnb.game.physics.medium import medium
from bnb.game.rules import POWER_MODES, STICK_LEVELS
from bnb.game.state import Fields, Ships

SCRATCH = pathlib.Path("/tmp/bnb-scratch-deltas")
TYPES = ("fighter", "heavy")
# The largest |nose rate| (rad/s) and |angle of attack| (rad) over both players and all
# presets (bench/state_deltas_measure.py's segments): scripted deathmatch and frontline.
OBSERVED_NOSE_RATE = 9.13
OBSERVED_ANGLE_OF_ATTACK = 0.27
OBSERVED_PROPER_SPEED = 130.3  # m/s, the heavy in scripted deathmatch_wide


def ships_of(ship_type, x, y, heading, speed, angle, nose_rate, energy, fields) -> Ships:
    """Living ships at the given state, the medium cached at their positions."""
    zeros = torch.zeros_like(x)
    log_index, gradient_x, gradient_y = medium(fields, x[None], y[None])
    return Ships(
        x=x,
        y=y,
        heading_cos=torch.cos(heading),
        heading_sin=torch.sin(heading),
        proper_speed=speed,
        angle_of_attack=angle,
        angle_of_attack_cos=torch.cos(angle),
        angle_of_attack_sin=torch.sin(angle),
        nose_rate=nose_rate,
        energy=energy,
        log_refractive_index=log_index[0],
        log_index_gradient_x=gradient_x[0],
        log_index_gradient_y=gradient_y[0],
        shield=zeros,
        shield_recharge_wait=zeros,
        alive=torch.ones_like(x, dtype=torch.bool),
        fired=torch.zeros_like(x, dtype=torch.bool),
        spawned=torch.zeros_like(x, dtype=torch.bool),
        respawn_wait=torch.zeros_like(x, dtype=torch.int32),
        gun_cooldown=torch.zeros_like(x, dtype=torch.int32),
        next_slot=torch.zeros_like(x, dtype=torch.int32),
        ship_type=ship_type,
    )


def no_fields(count: int = 1) -> Fields:
    """Fields far from every ship: a medium of ln n = 0."""
    far = torch.full((1, count), 1e7)
    return Fields(
        x=far,
        y=far,
        radius=torch.ones(1, count),
        band_width=torch.full((1, count), 40.0),
        level=torch.ones(1, count, dtype=torch.int32),
    )


def fly(rules, types, ships, stick, power, fields, ticks, hold_energy=False, switch=None):
    """Fly `ticks` ticks; returns the largest proper speed and |nose rate| seen per ship."""
    airframe = airframe_of(types, ships.ship_type)
    top_speed = ships.proper_speed.clone()
    top_rate = ships.nose_rate.abs()
    for tick in range(ticks):
        if switch is not None and tick % switch == 0 and tick > 0:
            stick = -stick
        ships = flight_tick(rules.flight, airframe, rules.tick_length, ships, stick, power, fields)
        if hold_energy:
            ships = dataclasses.replace(ships, energy=airframe.max_energy.clone())
        top_speed = torch.maximum(top_speed, ships.proper_speed)
        top_rate = torch.maximum(top_rate, ships.nose_rate.abs())
    return top_speed, top_rate


def envelope(rules, types) -> dict:
    """The terminal proper speeds and corner speed of each type, flown with flight_tick."""
    out = {}
    for t, name in enumerate(TYPES):
        n = 1
        ship_type = torch.full((n,), t)
        max_energy = float(types.max_energy[t])
        start = ships_of(ship_type, torch.zeros(n), torch.zeros(n), torch.zeros(n),
                         torch.full((n,), 30.0), torch.zeros(n), torch.zeros(n),
                         torch.full((n,), max_energy), no_fields())  # fmt: skip
        boost = torch.full((n,), POWER_MODES[1])
        coast = torch.full((n,), POWER_MODES[0])
        speed_boost, _ = fly(rules, types, start, torch.zeros(n), boost, no_fields(), 6000, True)
        speed_coast, _ = fly(rules, types, start, torch.zeros(n), coast, no_fields(), 6000)
        # the largest nose rate: full-stick reversals every k ticks at speeds up to terminal
        speeds = torch.linspace(5.0, float(speed_boost), 40)
        rates = []
        for switch in (1, 2, 3, 5, 8, 15, 30):
            ships = ships_of(torch.full((40,), t), torch.zeros(40), torch.zeros(40),
                             torch.zeros(40), speeds, torch.zeros(40), torch.zeros(40),
                             torch.full((40,), max_energy), no_fields())  # fmt: skip
            for power in POWER_MODES:
                _, rate = fly(rules, types, ships, torch.full((40,), 1.0),
                              torch.full((40,), power), no_fields(), 120, True, switch)  # fmt: skip
                rates.append(float(rate.max()))
        out[name] = dict(
            terminal_speed_boost_full_store=float(speed_boost),
            terminal_speed_coast=float(speed_coast),
            corner_speed=math.sqrt(
                float(types.max_acceleration[t] / types.max_lift_coefficient[t])
            ),
            nose_rate_full_stick_reversals_no_field=max(rates),
        )
    return out


def wrap(angle):
    """An angle wrapped to [-pi, pi)."""
    return torch.remainder(angle + math.pi, 2 * math.pi) - math.pi


@torch.no_grad()
def sweep(
    rules, types, env: dict, count: int = 30_000, seed: int = 0, speed_top: float | None = None
) -> dict:
    """The box's largest changes per type; speed_top replaces the terminal speed as the
    box's top proper speed."""
    generator = torch.Generator().manual_seed(seed)

    def uniform(low, high):
        return low + (high - low) * torch.rand(count, generator=generator)

    dt = rules.tick_length
    result = {}
    for t, name in enumerate(TYPES):
        u_top = speed_top or env[name]["terminal_speed_boost_full_store"]
        w_top = 1.25 * OBSERVED_NOSE_RATE
        a_top = 1.25 * OBSERVED_ANGLE_OF_ATTACK
        max_energy = float(types.max_energy[t])
        top = {}
        pairs = [(a, b) for a in (-2, -1, 1, 2) for b in (-2, -1, 1, 2)]
        for first, second in pairs:
            fields = Fields(x=torch.tensor([[0.0, 380.0]]), y=torch.zeros(1, 2),
                            radius=torch.full((1, 2), 200.0),
                            band_width=torch.full((1, 2), rules.field_band_width),
                            level=torch.tensor([[first, second]], dtype=torch.int32))  # fmt: skip
            for stick_index in range(len(STICK_LEVELS)):
                for power_index in range(len(POWER_MODES)):
                    distance, bearing = (
                        uniform(0.0, 300.0) ** 0.5 * 300.0**0.5,
                        uniform(-math.pi, math.pi),
                    )
                    x = 190.0 + distance * torch.cos(bearing)
                    y = distance * torch.sin(bearing)
                    ships = ships_of(torch.full((count,), t), x, y, uniform(-math.pi, math.pi),
                                     uniform(rules.flight.speed_floor, u_top),
                                     uniform(-a_top, a_top), uniform(-w_top, w_top),
                                     uniform(0.0, max_energy), fields)  # fmt: skip
                    stick, power = controls(torch.full((count,), power_index),
                                            torch.full((count,), stick_index))  # fmt: skip
                    after = flight_tick(rules.flight, airframe_of(types, ships.ship_type), dt,
                                        ships, stick, power, fields)  # fmt: skip
                    changes = changes_of(ships, after, dt, max_energy)
                    for key, value in changes.items():
                        top[key] = max(top.get(key, 0.0), float(value.abs().max()))
        result[name] = top
    return result


def world_velocity(ships: Ships):
    """(v_x, v_y, heading): the proper speed over n, along heading less the attack angle."""
    speed = ships.proper_speed / torch.exp(ships.log_refractive_index)
    heading = torch.atan2(ships.heading_sin, ships.heading_cos)
    direction = heading - torch.atan2(ships.angle_of_attack_sin, ships.angle_of_attack_cos)
    return speed * torch.cos(direction), speed * torch.sin(direction), heading


def changes_of(before: Ships, after: Ships, dt: float, max_energy: float) -> dict:
    """Each quantity's one-tick change between two ship states, by baseline."""
    vx0, vy0, h0 = world_velocity(before)
    vx1, vy1, h1 = world_velocity(after)
    dx, dy = after.x - before.x, after.y - before.y
    w = before.nose_rate
    half = 0.5 * w * dt
    sinc = torch.where(half.abs() > 1e-9, torch.sin(half) / torch.where(half == 0, 1, half), 1.0)
    ax = (torch.cos(half) * vx0 - torch.sin(half) * vy0) * dt * sinc
    ay = (torch.sin(half) * vx0 + torch.cos(half) * vy0) * dt * sinc
    rx = torch.cos(w * dt) * vx0 - torch.sin(w * dt) * vy0
    ry = torch.sin(w * dt) * vx0 + torch.cos(w * dt) * vy0
    dln = after.log_refractive_index - before.log_refractive_index
    predicted = (before.log_index_gradient_x * vx0 + before.log_index_gradient_y * vy0) * dt
    return {
        "position.persistence": torch.hypot(dx, dy),
        "position.dead_reckoning": torch.hypot(dx - vx0 * dt, dy - vy0 * dt),
        "position.arc": torch.hypot(dx - ax, dy - ay),
        "velocity.persistence": torch.hypot(vx1 - vx0, vy1 - vy0),
        "velocity.rotated": torch.hypot(vx1 - rx, vy1 - ry),
        "heading.persistence": wrap(h1 - h0),
        "heading.dead_reckoning": wrap(h1 - h0 - w * dt),
        "nose_rate.persistence": after.nose_rate - w,
        "angle_of_attack.persistence": after.angle_of_attack - before.angle_of_attack,
        "energy.persistence": (after.energy - before.energy) / max_energy,
        "log_refractive_index.persistence": dln,
        "log_refractive_index.dead_reckoning": dln - predicted,
        "speed_world": torch.hypot(vx1, vy1),
    }


def discrete(rules, types) -> dict:
    """The rules' bounds of the quantities that change in steps (per type, fractions of
    the maximum for shield and energy)."""
    dt = rules.tick_length
    out = {}
    for t, name in enumerate(TYPES):
        max_shield = float(types.max_shield[t])
        interval = round(1.0 / (dt * float(types.fire_rate[t])))
        out[name] = dict(
            shield_recharge_per_tick=rules.shield_recharge_rate * dt / max_shield,
            shield_largest_loss=1.0,  # any damage clamps at zero: the whole shield
            hit_damage_fraction_per_bullet=[
                float(types.hit_damage[k]) / max_shield for k in range(len(TYPES))
            ],  # fmt: skip
            # game v15: boundary_max_damage x smootherstep(depth / ramp), 0 at the line
            boundary_damage_per_tick_max=rules.boundary_max_damage * dt / max_shield,
            boundary_ramp_width_at_map_scale_1=rules.boundary_ramp_width,
            recharge_wait_steps=[-dt, rules.shield_recharge_delay],  # countdown, reset on a hit
            gun_cooldown_after_shot_ticks=interval - 1,  # set to the interval, then the clock
            bullet_energy_fraction=float(types.bullet_energy_cost[t] / types.max_energy[t]),
        )
    return out


def main() -> None:
    """Write the envelope, the two sweeps and the rules' discrete bounds to bounds.json."""
    torch.set_num_threads(1)
    rules = api.rules_of(api.PRESETS["deathmatch"])
    types = api.ship_types()
    env = envelope(rules, types)
    print(json.dumps(env, indent=1), flush=True)
    bounds = sweep(rules, types, env)
    observed_box = sweep(rules, types, env, speed_top=1.25 * OBSERVED_PROPER_SPEED)
    stall = rules.flight.stall_angle
    log_index_gradient = 2 * math.log(math.sqrt(2.0)) * 1.875 / rules.field_band_width
    report = dict(
        envelope=env,
        sweep=bounds,
        sweep_observed_speeds=observed_box,
        discrete=discrete(rules, types),
        analytic=dict(
            steepest_log_index_gradient_per_m=log_index_gradient,  # level 2, one field
            log_index_range=[-math.log(2.0), math.log(2.0)],
            stall_angle=stall,
            tick_length=rules.tick_length,
        ),
    )
    (SCRATCH / "bounds.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(bounds, indent=1))


if __name__ == "__main__":
    main()
