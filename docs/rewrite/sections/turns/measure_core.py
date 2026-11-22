"""Standard manoeuvres flown in both flight models, one ship, no fields: a table for the author.

    OMP_NUM_THREADS=1 PYTHONPATH=.:src uv run --no-sync python docs/rewrite/sections/turns/measure_core.py

Prints a markdown table (paste into docs/rewrite/core-flight-measurements.md).
"""

import math
import sys
from types import SimpleNamespace

import torch

from lean.game import card, core

torch.set_num_threads(1)
DEGREES = 180.0 / math.pi
COAST, DIVE, CLIMB = 0, 1, 2
FULL, SUSTAINED, FINE, STRAIGHT = 6, 4, 2, 0
LINEUP = tuple(card.TYPE_NAMES)  # every type, so a ship is flown by its card index
CONFIGS = {"v2": core.GameConfig(lineup=LINEUP), "core": core.GameConfig.core(lineup=LINEUP)}


class Ship:
    def __init__(self, model, ship_type, speed, store_fraction):
        self.model, self.config = model, CONFIGS[model]
        self.c = core.ship_constants(self.config, torch.full((1, 1), ship_type))
        z = torch.zeros(1, 1)
        self.s = SimpleNamespace(
            x=z.clone(), y=z.clone(), heading_cos=torch.ones(1, 1), heading_sin=z.clone(),
            speed=torch.full((1, 1), float(speed)), angle=z.clone(), angle_cos=torch.ones(1, 1),
            angle_sin=z.clone(), nose_rate=z.clone(), energy=store_fraction * torch.ones(1, 1) * self.c["max_energy"],
            log_index=z.clone(), gradient_x=z.clone(), gradient_y=z.clone(),
        )
        self.time = 0.0

    @property
    def speed(self):
        return self.s.speed.item()

    @property
    def store(self):
        return (self.s.energy / self.c["max_energy"]).item()

    def track(self):
        return math.atan2(self.s.heading_sin.item(), self.s.heading_cos.item()) - self.s.angle.item()

    def fly(self, stick, power, seconds, thrust_scale=1.0, until=None):
        """Fly; returns degrees the velocity turned. `until(ship)` stops early."""
        stick_t = torch.full((1, 1), stick, dtype=torch.long)
        power_t = torch.full((1, 1), power, dtype=torch.long)
        turned, previous = 0.0, self.track()
        with torch.no_grad():
            for _ in range(round(seconds / self.config.tick_length)):
                self.s = core.flight_tick(self.c, torch.zeros(1, 5, 0), self.config.tick_length, self.s, stick_t, power_t,
                                          model=self.model, thrust_scale=thrust_scale)
                self.time += self.config.tick_length
                now = self.track()
                turned += (now - previous + math.pi) % (2 * math.pi) - math.pi
                previous = now
                if until is not None and until(self):
                    break
        return abs(turned) * DEGREES


def corner_speed(ship_type):
    lift, limit = card.CARD["max_lift_coefficient"][ship_type], card.CARD["max_acceleration"][ship_type]
    return math.sqrt(limit / lift)


def measure(model, ship_type=2):
    rows = {}
    top = Ship(model, ship_type, 200.0, 0.5)
    top.fly(STRAIGHT, COAST, 120.0)
    rows["top level speed, coast (m/s)"] = f"{top.speed:.0f}"
    rows["store after 120 s straight coast (%)"] = f"{100 * top.store:.1f}"
    corner = corner_speed(ship_type)
    rows["corner speed (m/s, g limit meets lift)"] = f"{corner:.0f}"
    for name, stick in (("full", FULL), ("sustained", SUSTAINED), ("fine", FINE)):
        ship = Ship(model, ship_type, corner, 0.5)
        first = ship.fly(stick, COAST, 1.0)
        ship.fly(stick, COAST, 29.0)
        settled = ship.fly(stick, COAST, 1.0)
        rows[f"{name} stick from corner, coast: first second (deg)"] = f"{first:.0f}"
        rows[f"{name} stick, settled after 30 s: speed (m/s), rate (deg/s), store (%)"] = (
            f"{ship.speed:.0f}, {settled:.1f}, {100 * ship.store:.1f}")
    ship = Ship(model, ship_type, corner, 0.5)
    first = ship.fly(FULL, DIVE, 1.0)
    rows["full stick and dive from corner: first second (deg)"] = f"{first:.0f}"
    ship = Ship(model, ship_type, top.speed, 0.5)
    ship.fly(STRAIGHT, CLIMB, 60.0, until=lambda s: s.store > 0.99)
    rows["climb from top speed, half store: time to 99% (s), speed then (m/s)"] = (
        f"{ship.time:.1f}, {ship.speed:.0f}" if ship.store > 0.99 else f"not reached in 60 s ({100 * ship.store:.0f}%)")
    ship = Ship(model, ship_type, top.speed, 0.5)
    peak = [ship.speed]
    ship.fly(STRAIGHT, DIVE, 60.0, until=lambda s: peak.append(s.speed) or s.store < 0.01)
    rows["dive from top speed, half store: time to 1% (s), peak speed (m/s)"] = (
        f"{ship.time:.1f}, {max(peak):.0f}" if ship.store < 0.01 else f"not reached in 60 s ({100 * ship.store:.0f}%)")
    for label, speed in (("top speed", top.speed), ("corner speed", corner)):
        firing, quiet = Ship(model, ship_type, speed, 0.5), Ship(model, ship_type, speed, 0.5)
        scale = CONFIGS[model].fire_thrust_fraction if model == "core" else 1.0
        firing.fly(STRAIGHT, COAST, 1.0, thrust_scale=scale)
        quiet.fly(STRAIGHT, COAST, 1.0)
        rows[f"speed lost to 1 s of firing at {label} (m/s, against not firing)"] = f"{quiet.speed - firing.speed:.1f}"
    return rows


if __name__ == "__main__":
    ship_type = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    results = {model: measure(model, ship_type) for model in CONFIGS}
    print(f"| {card.TYPE_NAMES[ship_type]}, no fields | v2 | core |")
    print("|---|---|---|")
    for key in results["v2"]:
        print(f"| {key} | {results['v2'][key]} | {results['core'][key]} |")
