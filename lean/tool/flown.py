"""Flown checks of the doghouse: the game's own flight tick (lean.game.core.flight_tick, the flight
part of core.step), one ship per slot, no fields, for the design the card holds as type 0
(inside lean.game.tuning.flying, or the card as it stands). Game units at the card's time scale.

- `achievable_speeds`: per flight condition the doghouse draws (full, middle and gradual stick
  coasting; full stick diving and climbing), the lowest speed at which a ship flown from the
  spawn state actually flies it: the turn it makes (the velocity's direction change per tick)
  within `tolerance` of the quasi-steady rate at that speed and store (the line's value), and
  for dive and climb the store inside the edge bands (outside them dive and climb fade to coast).
  The search: from the spawn (spawn speed, spawn store) one of the 12 (stick, power) pairs held
  for 0 to 30 s (a branch every 0.5 s), then the condition held 4 s. The bleed that reaches the
  lowest states is a climb (it fills the store and slows the ship toward its stall speed); a
  random search of many-segment sequences finds the same limits to about 2 m/s
  (lean/tool/NOTES.md).
- `critical_points`: the corner, the best sustained turn and the tightest sustained turn flown
  (see there).

These fly a thousand ticks (seconds of CPU): the tool runs them on demand and caches them per
design; the slider path never flies.
"""

import math
from types import SimpleNamespace

import numpy as np
import torch

from lean.game import card, core
from lean.tool import flight_config

COAST, DIVE, CLIMB = 0, 1, 2
STRAIGHT, GRADUAL, MIDDLE, FULL = 0, 2, 4, 6  # stick indices (card.STICK_LEVELS), one side
# The doghouse's lines: name -> (stick index, power index).
CONDITIONS = {
    "full": (FULL, COAST),
    "middle": (MIDDLE, COAST),
    "gradual": (GRADUAL, COAST),
    "dive": (FULL, DIVE),
    "climb": (FULL, CLIMB),
}
TOLERANCE = 0.05  # a measured turn within 5% (and 0.2 deg/s) of the quasi-steady rate
BLEED_SECONDS, BLEED_STEP, HOLD_SECONDS = 30.0, 0.5, 4.0
HOLD_GAIN = 0.01  # stick per m/s of speed error: a held point's feedback
CHECK_SECONDS, SETTLE_SECONDS, AVERAGE_SECONDS = 20.0, 1.0, 2.0


def _fresh(n: int, speed, energy: float) -> SimpleNamespace:
    z = torch.zeros(n, 1)
    return SimpleNamespace(
        x=z.clone(),
        y=z.clone(),
        heading_cos=torch.ones(n, 1),
        heading_sin=z.clone(),
        speed=torch.as_tensor(speed, dtype=torch.float32).expand(n).reshape(n, 1).clone(),
        angle=z.clone(),
        angle_cos=torch.ones(n, 1),
        angle_sin=z.clone(),
        nose_rate=z.clone(),
        energy=torch.full((n, 1), float(energy)),
        log_index=z.clone(),
        gradient_x=z.clone(),
        gradient_y=z.clone(),
    )


def _turn_rate(before, after, tick_length: float) -> torch.Tensor:
    """The velocity's turn over a tick (rad/s): its direction is the heading turned back by the
    angle of attack."""
    a = torch.atan2(before.heading_sin, before.heading_cos) - before.angle
    b = torch.atan2(after.heading_sin, after.heading_cos) - after.angle
    return ((b - a + math.pi) % (2 * math.pi) - math.pi).abs()[:, 0] / tick_length


def _direction(s) -> SimpleNamespace:
    return SimpleNamespace(
        heading_cos=s.heading_cos.clone(), heading_sin=s.heading_sin.clone(), angle=s.angle.clone()
    )


def _ships(n: int):
    config = flight_config()
    constants = core.ship_constants(config, torch.zeros((n, 1), dtype=torch.long))
    p = core.per_type_constants(config, torch.zeros(n, dtype=torch.long))
    return config, constants, p


def achievable_speeds(tolerance: float = TOLERANCE) -> dict[str, float]:
    """Per condition (CONDITIONS), the lowest speed (game m/s) at which it is flown (the module
    docstring); inf if never."""
    pairs = [
        (stick, power)
        for stick in (STRAIGHT, GRADUAL, MIDDLE, FULL)
        for power in (COAST, DIVE, CLIMB)
    ]
    bleeds = np.arange(0.0, BLEED_SECONDS + 1e-9, BLEED_STEP)
    names = list(CONDITIONS)
    grid = [
        (c, b, t) for c in range(len(names)) for b in range(len(pairs)) for t in bleeds
    ]  # one ship each
    n = len(grid)
    config, constants, p = _ships(n)
    h = config.tick_length
    which = torch.tensor([g[0] for g in grid])
    condition = torch.tensor([CONDITIONS[names[g[0]]] for g in grid])
    bleed = torch.tensor([pairs[g[1]] for g in grid])
    switch = torch.tensor([round(g[2] / h) for g in grid])
    max_energy = float(p.max_energy[0])
    s = _fresh(n, card.SPAWN_SPEED, float(p.spawn_energy[0]))
    band = config.store_edge_band
    level, cap, deep = _shape(p, condition[:, :1])
    lowest = torch.full((len(names),), math.inf)
    fields = torch.zeros(n, 5, 0)
    with torch.no_grad():
        for tick in range(round((BLEED_SECONDS + HOLD_SECONDS) / h)):
            flying = tick >= switch
            control = torch.where(flying[:, None], condition, bleed)
            before = _direction(s)
            s = core.flight_tick(constants, fields, h, s, control[:, :1], control[:, 1:])
            measured = _turn_rate(before, s, h)
            speed, energy = s.speed[:, 0], s.energy[:, 0]
            quasi = torch.zeros(n)
            for power in (COAST, DIVE, CLIMB):
                quasi = torch.where(
                    condition[:, 1] == power,
                    core.steady_rates(p, speed, level, power, energy, cap, deep)[0],
                    quasi,
                )
            store = energy / max_energy
            inside = (store > band) & (store < 1.0 - band)
            ok = flying & (measured - quasi).abs().le(tolerance * quasi.abs() + math.radians(0.2))
            ok &= torch.where(condition[:, 1] != COAST, inside, True)
            here = torch.full_like(lowest, math.inf).scatter_reduce(
                0, which, torch.where(ok, speed, math.inf), "amin"
            )
            lowest = torch.minimum(lowest, here)
    return {name: float(v) for name, v in zip(names, lowest)}


def random_search(
    ships: int = 1024, seconds: float = 40.0, seed: int = 0, tolerance: float = TOLERANCE
) -> dict[str, float]:
    """The cross-check of achievable_speeds: from the spawn state, random sequences of the 12
    (stick, power) pairs held 0.2 to 12 s each (40% of segments a climb, the bleed); per
    condition the lowest speed at which it is flown by the same criterion. Never meaningfully
    below achievable_speeds if that search is complete (lean/tool/NOTES.md)."""
    generator = torch.Generator().manual_seed(seed)
    pairs = torch.tensor(
        [
            (stick, power)
            for stick in (STRAIGHT, GRADUAL, MIDDLE, FULL)
            for power in (COAST, DIVE, CLIMB)
        ]
    )
    climbs = torch.nonzero(pairs[:, 1] == CLIMB)[:, 0]

    def draw(k):
        any_pair = torch.randint(len(pairs), (k,), generator=generator)
        climb = climbs[torch.randint(len(climbs), (k,), generator=generator)]
        return torch.where(torch.rand(k, generator=generator) < 0.4, climb, any_pair)

    def duration(k):
        return 0.2 + 11.8 * torch.rand(k, generator=generator)

    config, constants, p = _ships(ships)
    h = config.tick_length
    max_energy = float(p.max_energy[0])
    band = config.store_edge_band
    s = _fresh(ships, card.SPAWN_SPEED, float(p.spawn_energy[0]))
    chosen, left = draw(ships), duration(ships)
    lowest = {name: math.inf for name in CONDITIONS}
    fields = torch.zeros(ships, 5, 0)
    with torch.no_grad():
        for _ in range(round(seconds / h)):
            control = pairs[chosen]
            before = _direction(s)
            s = core.flight_tick(constants, fields, h, s, control[:, :1], control[:, 1:])
            measured = _turn_rate(before, s, h)
            speed, energy = s.speed[:, 0], s.energy[:, 0]
            store = energy / max_energy
            level, cap, deep = _shape(p, control[:, :1])
            for name, (stick, power) in CONDITIONS.items():
                mask = (control[:, 0] == stick) & (control[:, 1] == power)
                if power != COAST:
                    mask &= (store > band) & (store < 1.0 - band)
                if not mask.any():
                    continue
                quasi = core.steady_rates(p, speed, level, power, energy, cap, deep)[0]
                mask &= (measured - quasi).abs().le(tolerance * quasi.abs() + math.radians(0.2))
                if mask.any():
                    lowest[name] = min(lowest[name], float(speed[mask].min()))
            left -= h
            change = left <= 0.0
            if change.any():
                k = int(change.sum())
                chosen[change], left[change] = draw(k), duration(k)
    return lowest


def _shape(p, stick: torch.Tensor):
    """Per ship, a stick index's (size, g cap, deep) as core.steady_rates takes them."""
    return tuple(
        table.gather(1, stick)[:, 0].abs()
        for table in (p.stick_table, p.stick_cap_table, p.deep_stall_table)
    )


def _holding_stick(p, speed: float, energy: float) -> float:
    """The stick (0 to 1) whose quasi-steady turn holds energy (Ps = 0) at `speed`, coasting; 1 if
    full stick gains."""
    v, e = torch.tensor([speed]), torch.tensor([energy])
    if float(core.quasi_steady(p, v, torch.ones(1), COAST, e)[2]) >= 0.0:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(40):
        middle = 0.5 * (low + high)
        if float(core.quasi_steady(p, v, torch.tensor([middle]), COAST, e)[2]) > 0.0:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def critical_points(
    corner_speed: float, best_speed: float | None, tightest_speed: float | None
) -> dict:
    """The doghouse's critical points flown (game units, rates in rad/s) from the quasi-steady
    speeds, at the spawn store, coasting, in one flight of a few ships for 20 s:

    - corner: full stick from 1.2, 1.4 and 1.7 x the corner speed, decelerating through it; the
      peak turn after the first second (the nose's transient).
    - best_sustained, tightest_sustained, where some stick below full holds energy at the
      point's speed ("held"): a ship started there holds it with that stick plus a feedback on
      the speed error (HOLD_GAIN); the mean speed and turn over the last 2 s. A second ship
      holds the same stick fixed: `fixed_stick_speed`, where it settles.
    - where full stick gains energy at the point's speed ("passed"; the slow side of the
      sustained curve, where Ps > 0 at full stick): the speed cannot be held (coasting, full
      stick accelerates the ship toward its settled full-stick speed), so full stick from 0.85,
      0.9 and 0.95 x the point's speed; after the first second, the largest turn (best) or the
      smallest radius (tightest) passed through.

    Each point: {speed, rate, radius, mode, stick, fixed_stick_speed}; None where the quasi-steady
    point is None. A held stick is the game's middle-stick entry set per ship each tick (a
    continuous stick; the agent's actions have three levels)."""
    _, _, p = _ships(1)
    energy = float(p.spawn_energy[0])
    starts, sticks, targets, gains, groups = [], [], [], [], []

    def add(group, speed, stick, target=0.0, gain=0.0):
        starts.append(speed)
        sticks.append(stick)
        targets.append(target)
        gains.append(gain)
        groups.append(group)

    for factor in (1.2, 1.4, 1.7):
        add("corner", factor * corner_speed, 1.0)
    modes = {}
    for name, speed in (("best_sustained", best_speed), ("tightest_sustained", tightest_speed)):
        if speed is None:
            continue
        stick = _holding_stick(p, speed, energy)
        if stick < 1.0 - 1e-6:
            modes[name] = ("held", stick)
            add(name, speed, stick, speed, HOLD_GAIN)
            add(name + " fixed", speed, stick)
        else:
            modes[name] = ("passed", 1.0)
            for factor in (0.85, 0.9, 0.95):
                add(name, factor * speed, 1.0)
    n = len(starts)
    config, constants, _ = _ships(n)
    h = config.tick_length
    s = _fresh(n, torch.tensor(starts), energy)
    control = torch.full((n, 1), MIDDLE)
    power = torch.full((n, 1), COAST)
    column = constants.columns["stick_table"][0] + MIDDLE
    # The held stick is a fraction of the envelope's top line: no g cap, no deep angle.
    constants.table[:, constants.columns["stick_cap_table"][0] + MIDDLE, 0] = 1.0
    constants.table[:, constants.columns["deep_stall_table"][0] + MIDDLE, 0] = 0.0
    base, target, gain = torch.tensor(sticks), torch.tensor(targets), torch.tensor(gains)
    fields = torch.zeros(n, 5, 0)
    speeds, rates = [], []
    with torch.no_grad():
        for _ in range(round(CHECK_SECONDS / h)):
            constants.table[:, column, 0] = (base + gain * (s.speed[:, 0] - target)).clamp(0.0, 1.0)
            before = _direction(s)
            s = core.flight_tick(constants, fields, h, s, control, power)
            rates.append(_turn_rate(before, s, h).numpy())
            speeds.append(s.speed[:, 0].numpy().copy())
    speeds, rates = np.array(speeds), np.array(rates)
    settle, average = round(SETTLE_SECONDS / h), round(AVERAGE_SECONDS / h)
    groups = np.array(groups)

    def extreme(group, largest_rate: bool):
        v, r = speeds[settle:, groups == group].ravel(), rates[settle:, groups == group].ravel()
        k = int(np.argmax(r)) if largest_rate else int(np.argmin(v / np.maximum(r, 1e-9)))
        return float(v[k]), float(r[k])

    v, rate = extreme("corner", True)
    out = {
        "corner": dict(speed=v, rate=rate, radius=v / rate),
        "best_sustained": None,
        "tightest_sustained": None,
    }
    for name, (mode, stick) in modes.items():
        if mode == "held":
            j = int(np.nonzero(groups == name)[0][0])
            v, rate = float(speeds[-average:, j].mean()), float(rates[-average:, j].mean())
            fixed = float(speeds[-average:, groups == name + " fixed"].mean())
        else:
            v, rate = extreme(name, name == "best_sustained")
            fixed = None
        out[name] = dict(
            speed=v, rate=rate, radius=v / rate, mode=mode, stick=stick, fixed_stick_speed=fixed
        )
    return out
