"""Measure the flight envelope of the ship physics.

Every scenario flies one ship per environment through ``update_ships`` directly,
so nothing but the flight model is measured: no collisions, no boundary, no
controller. On the field path (the live game's path, since Frontline always has
fields) each ship sits at the centre of one large uniform field of index n and
is re-centred after every tick, so it never meets an interface. The ambient
path takes n = 1 only.

Speeds are proper speeds ``u = n|v|`` unless a key says ``world``. Angles are
degrees and rates degrees per second. Lengths are world pixels.

``--baseline`` compares the settled envelope with an earlier artifact and
evaluates the slip model's validation checks (spec section 10.1).

Example (the checked-in slip-model measurement):
    uv run --no-sync python benchmarks/flight_envelope.py --device cuda \
        --baseline docs/internal/flight-envelope-baseline-oct2026.json \
        --out docs/internal/flight-envelope-slip-oct2026.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.config.core import EnvConfig
from boost_and_broadside.config.defaults import SHIP_CONFIG
from boost_and_broadside.constants import (
    NUM_POWER_ACTIONS,
    NUM_TURN_ACTIONS,
    PowerActions,
    ShootActions,
    TurnActions,
)
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.field_physics import refresh_ship_field_cache
from boost_and_broadside.env.frontline import frontline_ship_config
from boost_and_broadside.env.physics import (
    TURN_SHARP,
    TURN_SIDE,
    advance_bullets,
    slip_command,
    update_ships,
)

INDICES = (0.5, 2.0**-0.5, 1.0, 2.0**0.5, 2.0)
EM_SPEEDS = (10, 20, 30, 40, 60, 80, 100, 120, 136, 160, 200, 250, 300, 400, 500, 600)
ENTRY_SPEEDS = (40, 70, 100, 136, 180, 215, 300)
NOSE_SPEEDS = (40, 60, 100, 160, 215, 300)
# Renderer ship_size is 10 px centre to tip and collision_radius is 10 px.
SHIP_LENGTH_PX = 20.0
FIELD_TRANSITION_PX = 40.0

# "boost_unlimited" refills power every tick: not a game state, but the bound
# a ship approaches while it still has power to spend.
POWER_MODES = {
    "coast": (PowerActions.COAST, False),
    "boost": (PowerActions.BOOST, False),
    "boost_unlimited": (PowerActions.BOOST, True),
}


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


@dataclass
class Trace:
    """Per-tick history of a probe run, one column per scenario."""

    proper_speed: torch.Tensor  # (T+1, B)
    power: torch.Tensor  # (T+1, B)
    heading: torch.Tensor  # (T+1, B) cumulative path heading change, degrees
    gun: torch.Tensor  # (T+1, B) cumulative attitude change, degrees
    slip: torch.Tensor  # (T+1, B) nose angle to the flight path, degrees
    position: torch.Tensor  # (T+1, B) complex, displacement from start, world px
    dt: float


class FlightProbe:
    """Independent single ships, one per environment, in a uniform medium."""

    def __init__(
        self,
        config: ShipConfig,
        index: torch.Tensor,  # (B,)
        device: torch.device,
        *,
        path: str = "field",
        max_bullets: int = 0,
    ) -> None:
        if path not in {"field", "ambient"}:
            raise ValueError("path must be 'field' or 'ambient'")
        if path == "ambient" and not bool((index == 1.0).all()):
            raise ValueError("the ambient path has n = 1 only")
        self.config = config
        self.device = device
        self.index = index.to(device=device, dtype=torch.float32)  # (B,)
        batch = index.shape[0]
        env_config = EnvConfig(
            num_ships=1,
            max_bullets=max_bullets,
            max_episode_steps=None,
            num_fields=1 if path == "field" else 0,
        )
        env = TensorEnv(batch, config, env_config, device)
        env.reset(seed=0)
        self.state = env.state
        world_w, world_h = config.world_size
        self.centre = complex(world_w / 2.0, world_h / 2.0)
        state = self.state
        state.ship_pos = torch.full_like(state.ship_pos, self.centre)  # (B, 1)
        state.ship_alive = torch.ones_like(state.ship_alive)
        if path == "field":
            # The flat core reaches well past one tick of travel, and the ship is
            # re-centred every tick, so it only ever samples the interior index.
            radius = 0.4 * min(config.world_size)
            state.field_pos = torch.full_like(state.field_pos, self.centre)  # (B, 1)
            state.field_radius = torch.full_like(state.field_radius, radius)
            state.field_transition_width = torch.full_like(
                state.field_transition_width, FIELD_TRANSITION_PX
            )
            state.field_index = self.index.unsqueeze(1)  # (B, 1)
            refresh_ship_field_cache(state, config)

    def set_flight(self, proper_speed: torch.Tensor, power: torch.Tensor) -> None:
        """Fly straight along +x at the given proper speed and power."""
        state = self.state
        speed = proper_speed.to(self.device, torch.float32) / self.index  # (B,) world
        state.ship_vel = torch.complex(speed, torch.zeros_like(speed)).unsqueeze(1)  # (B, 1)
        state.ship_attitude = torch.ones_like(state.ship_attitude)
        state.ship_ang_vel = torch.zeros_like(state.ship_ang_vel)
        state.ship_slip = torch.zeros_like(state.ship_slip)
        state.ship_slip_rate = torch.zeros_like(state.ship_slip_rate)
        state.ship_power = power.to(self.device, torch.float32).unsqueeze(1)  # (B, 1)
        state.ship_cooldown = torch.zeros_like(state.ship_cooldown)

    @property
    def proper_speed(self) -> torch.Tensor:
        return (self.state.ship_local_index * self.state.ship_vel.abs())[:, 0]  # (B,)

    @property
    def slip(self) -> torch.Tensor:
        return torch.rad2deg(self.state.ship_slip[:, 0])  # (B,)

    @property
    def power(self) -> torch.Tensor:
        return self.state.ship_power[:, 0]  # (B,)

    @property
    def energy(self) -> torch.Tensor:
        """Generalized energy ½u² + power_speed_constant·power."""
        return 0.5 * self.proper_speed.square() + self.config.power_speed_constant * self.power

    def step(
        self,
        power_action: torch.Tensor,  # (B,)
        turn_action: torch.Tensor,  # (B,)
        refill_power: torch.Tensor | None = None,  # (B,) bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance one tick and return (displacement, heading change, attitude change).

        Displacement is complex world px; both changes are radians.
        """
        state = self.state
        shoot_action = torch.full_like(power_action, int(ShootActions.NO_SHOOT))
        actions = torch.stack([power_action, turn_action, shoot_action], dim=-1)  # (B, 3)
        actions = actions.float().unsqueeze(1)  # (B, 1, 3)
        previous_vel = state.ship_vel[:, 0]  # (B,)
        previous_attitude = state.ship_attitude[:, 0]  # (B,)
        update_ships(state, actions, self.config)
        world_w, world_h = self.config.world_size
        offset = state.ship_pos[:, 0] - self.centre  # (B,) complex
        displacement = torch.complex(
            (offset.real + world_w / 2) % world_w - world_w / 2,
            (offset.imag + world_h / 2) % world_h - world_h / 2,
        )
        state.ship_pos = torch.full_like(state.ship_pos, self.centre)
        if refill_power is not None:
            state.ship_power = torch.where(
                refill_power.unsqueeze(1),
                torch.full_like(state.ship_power, self.config.max_power),
                state.ship_power,
            )
        heading_change = torch.angle(state.ship_vel[:, 0] * previous_vel.conj())  # (B,)
        attitude_change = torch.angle(state.ship_attitude[:, 0] * previous_attitude.conj())  # (B,)
        return displacement, heading_change, attitude_change


Controller = Callable[[int], tuple[torch.Tensor, torch.Tensor]]


def fly(
    probe: FlightProbe,
    ticks: int,
    controller: Controller,
    refill_power: torch.Tensor | None = None,  # (B,) bool
) -> Trace:
    """Fly ``ticks`` ticks under ``controller(tick) -> (power, turn)``."""
    batch = probe.index.shape[0]
    zeros = torch.zeros(batch, device=probe.device)
    speeds, powers, headings, guns, slips, positions = (
        [probe.proper_speed],
        [probe.power],
        [zeros],
        [zeros],
        [probe.slip],
        [torch.complex(zeros, zeros)],
    )
    for tick in range(ticks):
        power_action, turn_action = controller(tick)
        displacement, heading_change, attitude_change = probe.step(
            power_action, turn_action, refill_power
        )
        speeds.append(probe.proper_speed)
        powers.append(probe.power)
        headings.append(headings[-1] + torch.rad2deg(heading_change))
        guns.append(guns[-1] + torch.rad2deg(attitude_change))
        slips.append(probe.slip)
        positions.append(positions[-1] + displacement)
    return Trace(
        torch.stack(speeds).cpu(),  # (T+1, B)
        torch.stack(powers).cpu(),
        torch.stack(headings).cpu(),
        torch.stack(guns).cpu(),
        torch.stack(slips).cpu(),
        torch.stack(positions).cpu(),
        probe.config.dt,
    )


def constant(actions: Sequence[tuple[int, int]], device: torch.device) -> Controller:
    """A controller that holds one (power, turn) pair per scenario."""
    power = torch.tensor([p for p, _ in actions], device=device)  # (B,)
    turn = torch.tensor([t for _, t in actions], device=device)  # (B,)
    return lambda tick: (power, turn)


def first_tick(mask: torch.Tensor) -> list[int | None]:
    """Index of the first True along time for each column of a (T+1, B) mask."""
    hit = mask.any(dim=0)  # (B,)
    first = mask.float().argmax(dim=0)  # (B,)
    return [int(f) if h else None for f, h in zip(first.tolist(), hit.tolist())]


def _seconds(tick: int | None, dt: float) -> float | None:
    return None if tick is None else round(tick * dt, 4)


def _r(value: float, digits: int = 3) -> float | None:
    return None if value is None or not math.isfinite(value) else round(value, digits)


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


def measure_terminal(
    config: ShipConfig,
    device: torch.device,
    *,
    path: str = "field",
    indices: Sequence[float] = INDICES,
    seconds: float = 120.0,
    window_seconds: float = 10.0,
) -> list[dict]:
    """Steady state of every held (power mode, turn) pair from cruise at full power.

    Turn rate and radius come from the last window; ``drift`` compares that
    window's mean speed with the one before it, as a convergence check.
    """
    combos = list(itertools.product(indices, POWER_MODES, TurnActions))
    index = torch.tensor([n for n, _, _ in combos])
    probe = FlightProbe(config, index, device, path=path)
    probe.set_flight(
        torch.full((len(combos),), 100.0), torch.full((len(combos),), config.max_power)
    )
    refill = torch.tensor([POWER_MODES[m][1] for _, m, _ in combos], device=device)
    controller = constant([(POWER_MODES[m][0], t) for _, m, t in combos], device)
    ticks = round(seconds / config.dt)
    trace = fly(probe, ticks, controller, refill)

    window = round(window_seconds / config.dt)
    last = trace.proper_speed[-window:].mean(0)  # (B,)
    before = trace.proper_speed[-2 * window : -window].mean(0)  # (B,)
    heading_rate = (trace.heading[-1] - trace.heading[-1 - window]) / window_seconds  # (B,)
    steps = trace.position[-window:] - trace.position[-window - 1 : -1]  # (W, B)
    world_distance = steps.abs().sum(0)  # (B,)
    turned = torch.deg2rad((trace.heading[-1] - trace.heading[-1 - window]).abs())  # (B,)

    rows = []
    for column, (n, mode, turn) in enumerate(combos):
        rate = heading_rate[column].item()
        radius = (world_distance[column] / turned[column]).item() if turned[column] > 1e-3 else None
        rows.append(
            {
                "index": round(n, 4),
                "power_mode": mode,
                "turn": turn.name,
                "proper_speed": _r(last[column].item()),
                "world_speed": _r(last[column].item() / n),
                "turn_rate_deg_s": _r(rate),
                "radius_world_px": _r(radius) if radius is not None else None,
                "power_end": _r(trace.power[-1, column].item()),
                "drift": _r(abs(last[column] - before[column]).item() / last[column].item(), 5),
            }
        )
    return rows


def measure_em_grid(
    config: ShipConfig,
    device: torch.device,
    *,
    path: str = "field",
    indices: Sequence[float] = INDICES,
    speeds: Sequence[float] = EM_SPEEDS,
    start_power: float = 50.0,
) -> list[dict]:
    """One-tick instantaneous performance: the energy-manoeuvrability grid.

    The nose starts settled at the slip the turn action commands at that speed,
    so the tick measures the turn the ship holds there rather than the onset.

    ``energy_rate`` is d(½u² + K·power)/dt, which includes passive regeneration.
    ``excess_power_accel`` divides it by u: the along-track acceleration the
    ship would have if the whole energy change went into speed, the analogue
    of an aircraft's specific excess power.
    """
    combos = list(itertools.product(indices, speeds, PowerActions, TurnActions))
    batch = len(combos)
    index = torch.tensor([n for n, _, _, _ in combos])
    probe = FlightProbe(config, index, device, path=path)
    speed = torch.tensor([float(u) for _, u, _, _ in combos])
    probe.set_flight(speed, torch.full((batch,), start_power))
    power = torch.tensor([p for _, _, p, _ in combos], device=device)
    turn = torch.tensor([t for _, _, _, t in combos], device=device)
    side = torch.tensor([TURN_SIDE[t] for _, _, _, t in combos], device=device)  # (B,)
    sharp = torch.tensor([TURN_SHARP[t] for _, _, _, t in combos], device=device)  # (B,)
    settled = slip_command(side, sharp, speed.to(device), config)  # (B,)
    probe.state.ship_slip = settled.unsqueeze(1)  # (B, 1)
    probe.state.ship_attitude = torch.polar(torch.ones_like(settled), settled).unsqueeze(1)
    energy_before = probe.energy
    speed_before = probe.proper_speed
    _, heading_change, _ = probe.step(power, turn)
    dt = config.dt
    rate = torch.rad2deg(heading_change) / dt  # (B,)
    world_speed = 0.5 * (speed_before + probe.proper_speed) / probe.index  # (B,)
    radius = world_speed / heading_change.abs().clamp(min=1e-9) * dt  # (B,)
    along = (probe.proper_speed - speed_before) / dt  # (B,)
    energy_rate = (probe.energy - energy_before) / dt  # (B,)
    lateral = torch.deg2rad(rate.abs()) * world_speed * probe.index  # (B,) proper

    rows = []
    for column, (n, u, p, t) in enumerate(combos):
        rows.append(
            {
                "index": round(n, 4),
                "proper_speed": u,
                "power": p.name,
                "turn": t.name,
                "turn_rate_deg_s": _r(rate[column].item()),
                "radius_world_px": _r(radius[column].item()) if t not in _STRAIGHT else None,
                "lateral_accel": _r(lateral[column].item()),
                "along_accel": _r(along[column].item()),
                "energy_rate": _r(energy_rate[column].item(), 1),
                "excess_power_accel": _r(energy_rate[column].item() / u),
            }
        )
    return rows


_STRAIGHT = {TurnActions.GO_STRAIGHT}


def measure_straight_line(
    config: ShipConfig,
    device: torch.device,
    *,
    path: str = "field",
    indices: Sequence[float] = INDICES,
    seconds: float = 60.0,
) -> list[dict]:
    """Acceleration, deceleration and boost endurance in straight flight."""
    scenarios = {
        # name: (start proper speed, start power, power action, turn action)
        "boost_from_rest": (2.0, config.max_power, PowerActions.BOOST, TurnActions.GO_STRAIGHT),
        "boost_from_cruise": (100.0, config.max_power, PowerActions.BOOST, TurnActions.GO_STRAIGHT),
        "coast_from_dash": (215.0, 0.0, PowerActions.COAST, TurnActions.GO_STRAIGHT),
        "sharp_pull_from_dash": (215.0, 0.0, PowerActions.COAST, TurnActions.SHARP_LEFT),
        "reverse_from_dash": (215.0, 0.0, PowerActions.REVERSE, TurnActions.GO_STRAIGHT),
        "reverse_sharp_pull_from_dash": (215.0, 0.0, PowerActions.REVERSE, TurnActions.SHARP_LEFT),
        "reverse_from_cruise": (100.0, 0.0, PowerActions.REVERSE, TurnActions.GO_STRAIGHT),
    }
    combos = list(itertools.product(indices, scenarios))
    index = torch.tensor([n for n, _ in combos])
    probe = FlightProbe(config, index, device, path=path)
    probe.set_flight(
        torch.tensor([scenarios[s][0] for _, s in combos]),
        torch.tensor([scenarios[s][1] for _, s in combos]),
    )
    controller = constant([scenarios[s][2:] for _, s in combos], device)
    trace = fly(probe, round(seconds / config.dt), controller)
    dt = config.dt

    rows = []
    for column, (n, name) in enumerate(combos):
        speed = trace.proper_speed[:, column]
        distance = trace.position[:, column].real
        start = scenarios[name][0]
        rising = scenarios[name][2] == PowerActions.BOOST
        thresholds = (25, 50, 100, 136, 150, 200, 250) if rising else (150, 100, 50, 25, 10, 2)
        crossings = {}
        for threshold in thresholds:
            if (threshold <= start) if rising else (threshold >= start):
                continue
            mask = (speed >= threshold) if rising else (speed <= threshold)
            tick = first_tick(mask.unsqueeze(1))[0]
            crossings[str(threshold)] = {
                "seconds": _seconds(tick, dt),
                "distance_world_px": _r(distance[tick].item(), 1) if tick is not None else None,
            }
        power_out = first_tick((trace.power[1:, column] <= 0.5).unsqueeze(1))[0]
        peak_tick = int(speed.argmax())
        rows.append(
            {
                "index": round(n, 4),
                "scenario": name,
                "start_speed": start,
                "crossings": crossings,
                "peak_speed": _r(speed.max().item()),
                "peak_seconds": _r(peak_tick * dt),
                "end_speed": _r(speed[-1].item()),
                "power_out_seconds": _seconds(power_out + 1, dt)
                if rising and power_out is not None
                else None,
                "power_end": _r(trace.power[-1, column].item()),
                "power_after_1s": _r(trace.power[round(1.0 / dt), column].item()),
            }
        )
    return rows


def measure_turn_transients(
    config: ShipConfig,
    device: torch.device,
    *,
    path: str = "field",
    indices: Sequence[float] = (1.0,),
    entry_speeds: Sequence[float] = ENTRY_SPEEDS,
    seconds: float = 8.0,
) -> list[dict]:
    """Time and space to turn 90°, 180° and 360° from straight flight at full power.

    ``advance_90`` is forward travel when the heading reaches 90°;
    ``diameter_180`` is the lateral offset when it reaches 180°.
    """
    turns = (TurnActions.TURN_LEFT, TurnActions.SHARP_LEFT)
    powers = (PowerActions.COAST, PowerActions.BOOST)
    combos = list(itertools.product(indices, entry_speeds, powers, turns))
    batch = len(combos)
    probe = FlightProbe(config, torch.tensor([n for n, _, _, _ in combos]), device, path=path)
    probe.set_flight(
        torch.tensor([float(u) for _, u, _, _ in combos]),
        torch.full((batch,), config.max_power),
    )
    controller = constant([(p, t) for _, _, p, t in combos], device)
    trace = fly(probe, round(seconds / config.dt), controller)
    dt = config.dt
    heading = trace.heading.abs()  # (T+1, B)
    t90, t180, t360 = (first_tick(heading >= angle) for angle in (90.0, 180.0, 360.0))

    rows = []
    for column, (n, u, p, t) in enumerate(combos):
        position = trace.position[:, column]
        rows.append(
            {
                "index": round(n, 4),
                "entry_speed": u,
                "power": p.name,
                "turn": t.name,
                "seconds_90": _seconds(t90[column], dt),
                "seconds_180": _seconds(t180[column], dt),
                "seconds_360": _seconds(t360[column], dt),
                "speed_at_180": _r(trace.proper_speed[t180[column], column].item())
                if t180[column] is not None
                else None,
                "advance_90_world_px": _r(position[t90[column]].real.item(), 1)
                if t90[column] is not None
                else None,
                "diameter_180_world_px": _r(abs(position[t180[column]].imag.item()), 1)
                if t180[column] is not None
                else None,
            }
        )
    return rows


def measure_nose(
    config: ShipConfig,
    device: torch.device,
    *,
    path: str = "field",
    indices: Sequence[float] = (1.0,),
    speeds: Sequence[float] = NOSE_SPEEDS,
    hold_seconds: float = 0.5,
) -> list[dict]:
    """Gun (attitude) response: a nose throw and a full reversal, coasting.

    ``throw`` holds sharp left from straight flight, then releases to straight.
    ``reversal`` holds sharp left, then switches to sharp right. Onset is the
    time for the slip to reach 90% of its settled value under the hold;
    ``slip_settle_seconds_after_switch`` is the time after the switch until
    the slip is within 1° of its new command (zero for a throw, 90% of the
    opposite for a reversal).
    """
    hold = round(hold_seconds / config.dt)
    combos = list(itertools.product(indices, ("throw", "reversal"), speeds))
    batch = len(combos)
    probe = FlightProbe(config, torch.tensor([n for n, _, _ in combos]), device, path=path)
    probe.set_flight(torch.tensor([float(u) for _, _, u in combos]), torch.full((batch,), 50.0))
    after = torch.tensor(
        [
            TurnActions.GO_STRAIGHT if kind == "throw" else TurnActions.SHARP_RIGHT
            for _, kind, _ in combos
        ],
        device=device,
    )
    coast = torch.full((batch,), int(PowerActions.COAST), device=device)
    left = torch.full((batch,), int(TurnActions.SHARP_LEFT), device=device)

    def controller(tick: int) -> tuple[torch.Tensor, torch.Tensor]:
        return coast, left if tick < hold else after

    trace = fly(probe, 2 * hold, controller)
    dt = config.dt
    sample_seconds = (dt, 0.1, 0.25, hold_seconds)
    rows = []
    for column, (n, kind, u) in enumerate(combos):
        gun = trace.gun[:, column]
        heading = trace.heading[:, column]
        slip = trace.slip[:, column]
        gun_rate = (gun[1:] - gun[:-1]).abs() / dt
        held = slip[: hold + 1]
        settled = held[-1].item()
        onset = first_tick((held.abs() >= 0.9 * abs(settled)).unsqueeze(1))[0]
        released = slip[hold + 1 :]
        target = 0.0 if kind == "throw" else -0.9 * settled
        if kind == "throw":
            reached = released.abs() <= 1.0
        else:
            reached = released * math.copysign(1.0, target) >= abs(target)
        settle = first_tick(reached.unsqueeze(1))[0]
        slip_rate = slip[1:] - slip[:-1]
        hold_rate = slip_rate[:hold]
        rows.append(
            {
                "index": round(n, 4),
                "kind": kind,
                "speed": u,
                "gun_deg": {f"{s:.3f}": _r(gun[round(s / dt)].item(), 2) for s in sample_seconds},
                "path_deg": {
                    f"{s:.3f}": _r(heading[round(s / dt)].item(), 2) for s in sample_seconds
                },
                "slip_deg": {f"{s:.3f}": _r(slip[round(s / dt)].item(), 2) for s in sample_seconds},
                "peak_gun_rate_deg_s": _r(gun_rate.max().item(), 1),
                "peak_slip_rate_deg_s": _r((slip_rate.abs().max() / dt).item(), 1),
                "max_abs_slip_deg": _r(slip.abs().max().item(), 3),
                "onset_90_seconds": _seconds(onset, dt),
                "hold_slip_rate_sign_changes": int(
                    ((hold_rate[1:] * hold_rate[:-1]) < -1e-9).sum().item()
                ),
                "gun_step_on_switch_deg": _r((gun[hold + 1] - gun[hold]).item(), 2),
                "slip_settle_seconds_after_switch": _seconds(
                    None if settle is None else settle + 1, dt
                ),
                "gun_end_deg": _r(gun[-1].item(), 2),
            }
        )
    return rows


def measure_bullets(
    config: ShipConfig,
    device: torch.device,
    *,
    ship_speeds: Sequence[float] = (0.0, 100.0, 215.0),
) -> list[dict]:
    """Range and time of flight of one forward shot from straight flight at n = 1."""
    config = replace(config, bullet_spread=0.0)
    batch = len(ship_speeds)
    probe = FlightProbe(config, torch.ones(batch), device, max_bullets=1)
    probe.set_flight(torch.tensor([float(u) for u in ship_speeds]), torch.full((batch,), 50.0))
    state = probe.state
    shoot = torch.full((batch,), int(ShootActions.SHOOT), device=device)
    straight = torch.zeros(batch, dtype=torch.long, device=device)
    actions = torch.stack([straight, straight, shoot], dim=-1).float().unsqueeze(1)  # (B, 1, 3)
    update_ships(state, actions, config)
    origin = state.bullet_pos[:, 0, 0]  # (B,)
    ranges, speeds = [torch.zeros(batch, device=device)], [state.bullet_vel[:, 0, 0].abs()]
    while bool(state.bullet_active.any()):
        state, _ = advance_bullets(state, config)
        ranges.append((state.bullet_pos[:, 0, 0] - origin).abs())
        speeds.append(state.bullet_vel[:, 0, 0].abs())
    ranges = torch.stack(ranges).cpu()  # (T+1, B)
    speeds = torch.stack(speeds).cpu()  # (T+1, B)
    dt = config.dt
    rows = []
    for column, u in enumerate(ship_speeds):
        rows.append(
            {
                "ship_speed": u,
                "muzzle_speed_world": _r(speeds[0, column].item()),
                "range_world_px": _r(ranges[-1, column].item(), 1),
                "end_speed_world": _r(speeds[-1, column].item()),
                "seconds_to": {
                    str(d): _seconds(first_tick((ranges[:, column] >= d).unsqueeze(1))[0], dt)
                    for d in (100, 200, 300, 400)
                },
            }
        )
    return rows


@torch.inference_mode()
def measure_play(device: torch.device, *, envs: int, decisions: int, seeds: Sequence[int]) -> dict:
    """How the stochastic scripted controller uses the envelope in live Frontline play."""
    from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
    from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
    from boost_and_broadside.env.perception import team_visibility_from_state
    from boost_and_broadside.profiles import PROFILES
    from boost_and_broadside.runtime.actions import (
        PendingActionState,
        advance_autonomous_decision,
    )

    profile = PROFILES["rl"]
    config = profile.ship_config
    env_config = EnvConfig(
        num_ships=profile.num_ships,
        num_fields=profile.num_fields,
        max_bullets=profile.max_bullets,
        max_episode_steps=profile.max_episode_steps,
        action_repeat=profile.action_repeat,
        spawn_resource_spread=profile.spawn_resource_spread,
        frontline=profile.frontline,
        vision_range=profile.vision_range,
        zones_occlude=profile.zones_occlude,
        spawn_reveal=profile.spawn_reveal,
    )
    decision_dt = config.dt * profile.action_repeat
    speeds, rates, indices, joint = [], [], [], []
    for seed in seeds:
        torch.manual_seed(seed)
        env = TensorEnv(envs, config, env_config, device)
        env.reset(seed=seed)
        scripted = StochasticScriptedAgent(config, StochasticAgentConfig())
        pending = PendingActionState.allocate(envs, profile.num_ships, device)
        for _ in range(decisions):
            before_vel = env.state.ship_vel
            sight = team_visibility_from_state(
                env.state, config, env_config, perceive_bullets=False
            )
            selected = scripted.get_actions(env.state, sight.ship)
            applied = pending.applied_action()  # (B, N, 3)
            done, truncated, _ = advance_autonomous_decision(env, pending, selected)
            state = env.state
            valid = state.ship_alive & ~state.ship_respawned  # (B, N)
            speed = state.ship_local_index * state.ship_vel.abs()  # (B, N)
            rate = torch.rad2deg(torch.angle(state.ship_vel * before_vel.conj())) / decision_dt
            speeds.append(speed[valid])
            rates.append(rate[valid])
            indices.append(state.ship_local_index[valid])
            code = applied[..., 0].long() * NUM_TURN_ACTIONS + applied[..., 1].long()
            joint.append(code[valid])
            finished = done | truncated
            if bool(finished.any()):
                env.reset_envs(finished)
                pending.reset(finished)

    speed = torch.cat(speeds).float().cpu()
    rate = torch.cat(rates).float().cpu()
    index = torch.cat(indices).float().cpu()
    joint = torch.cat(joint).cpu()
    quantiles = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
    q = torch.tensor(quantiles)

    def percentiles(values: torch.Tensor) -> dict[str, float]:
        return {
            f"p{100 * level:g}": _r(v)
            for level, v in zip(quantiles, torch.quantile(values, q).tolist())
        }

    lateral = torch.deg2rad(rate.abs()) * speed
    counts = torch.bincount(joint, minlength=NUM_POWER_ACTIONS * NUM_TURN_ACTIONS).float()
    counts = counts.view(NUM_POWER_ACTIONS, NUM_TURN_ACTIONS) / counts.sum()
    nearest = (index.unsqueeze(1) - torch.tensor(INDICES)).abs()  # (S, 5)
    distance, level = nearest.min(dim=1)
    on_level = distance < 0.01
    return {
        "controller": "StochasticScriptedAgent vs itself",
        "profile": "rl",
        "seeds": list(seeds),
        "envs": envs,
        "decisions_per_seed": decisions,
        "decision_hz": 1.0 / decision_dt,
        "ship_decisions": speed.numel(),
        "proper_speed": percentiles(speed),
        "fraction_above": {
            str(u): _r((speed > u).float().mean().item(), 4) for u in (100, 120, 136, 160, 215)
        },
        "abs_heading_rate_deg_s": percentiles(rate.abs()),
        "lateral_accel": percentiles(lateral),
        "action_fraction": {
            power.name: {turn.name: _r(counts[power, turn].item(), 4) for turn in TurnActions}
            for power in PowerActions
        },
        "local_index_fraction": {
            **{
                f"{n:.3f}": _r((on_level & (level == i)).float().mean().item(), 4)
                for i, n in enumerate(INDICES)
            },
            "transition": _r((~on_level).float().mean().item(), 4),
        },
    }


# Settled rows that stay below both corners, so the slip model must reproduce
# the earlier envelope: every coasting turn, the boosted sharp turn, and the
# straight-line terminals. The boosted normal turn and the power-held-full turns
# settle above corner and are meant to change.
_UNCHANGED_TERMINALS = {
    ("coast", "GO_STRAIGHT"),
    ("coast", "TURN_LEFT"),
    ("coast", "SHARP_LEFT"),
    ("boost", "GO_STRAIGHT"),
    ("boost", "SHARP_LEFT"),
    ("boost_unlimited", "GO_STRAIGHT"),
}


def _within(value: float | None, low: float, high: float) -> bool:
    return value is not None and low <= value <= high


def validate(artifact: dict, baseline: dict | None, config: ShipConfig) -> dict:
    """The slip model's physics checks (spec section 10.1), each with its numbers."""
    checks: dict[str, dict] = {}

    if baseline is not None:
        before = {
            (r["index"], r["power_mode"], r["turn"]): r
            for r in baseline["terminal"]
            if (r["power_mode"], r["turn"]) in _UNCHANGED_TERMINALS
        }
        worst = 0.0
        rows, above_corner = [], []
        for row in artifact["terminal"]:
            key = (row["index"], row["power_mode"], row["turn"])
            if key not in before:
                continue
            if row["turn"] != "GO_STRAIGHT" and row["proper_speed"] > config.corner_speed:
                # A second, g-limited equilibrium: entered fast with power to
                # spare, the turn never bleeds down to the stall-slip one.
                above_corner.append({"key": list(key), "proper_speed": row["proper_speed"]})
                continue
            old = before[key]
            speed_error = row["proper_speed"] / old["proper_speed"] - 1.0
            rate_error = (
                abs(row["turn_rate_deg_s"]) / abs(old["turn_rate_deg_s"]) - 1.0
                if abs(old["turn_rate_deg_s"] or 0.0) > 1e-6
                else 0.0
            )
            worst = max(worst, abs(speed_error), abs(rate_error))
            rows.append({"key": list(key), "speed": _r(speed_error, 4), "rate": _r(rate_error, 4)})
        checks["below_corner_matches_baseline"] = {
            "pass": worst <= 0.02,
            "worst_relative_error": _r(worst, 4),
            "rows": rows,
            "settled_above_corner": above_corner,
        }

    allowance = config.boost_thrust * math.sin(config.stall_angle)
    above = [
        r
        for r in artifact["em_grid"]
        if r["index"] == 1.0 and r["turn"] == "SHARP_LEFT" and r["proper_speed"] > 100
    ]
    peak = max(r["lateral_accel"] for r in above)
    checks["lateral_accel_above_corner"] = {
        "pass": peak <= config.max_lateral_accel + allowance,
        "peak": peak,
        "limit": config.max_lateral_accel,
        "slipped_thrust_allowance": _r(allowance),
    }

    nose = artifact["nose"]
    at_one = [r for r in nose if r["index"] == 1.0]

    # Step responses are read below corner, where the command is constant
    # while the speed bleeds; above it the command grows as the ship slows.
    def stepped(row: dict) -> bool:
        return config.slip_fade_speed <= row["speed"] <= config.corner_speed

    onsets = [r["onset_90_seconds"] for r in at_one if stepped(r)]
    snaps = [
        r["slip_settle_seconds_after_switch"] for r in at_one if r["kind"] == "throw" and stepped(r)
    ]
    reversals = [r["slip_settle_seconds_after_switch"] for r in at_one if r["kind"] == "reversal"]
    checks["onset_90"] = {"pass": all(_within(t, 0.13, 0.20) for t in onsets), "seconds": onsets}
    checks["snap_back"] = {"pass": all(_within(t, 0.13, 0.20) for t in snaps), "seconds": snaps}
    checks["reversal"] = {"seconds": reversals}
    checks["no_jitter"] = {
        "pass": all(r["hold_slip_rate_sign_changes"] == 0 for r in nose)
        and all(r["max_abs_slip_deg"] <= math.degrees(config.stall_angle) + 1e-3 for r in nose),
        "max_abs_slip_deg": max(r["max_abs_slip_deg"] for r in nose),
    }

    reference = {(r["kind"], r["speed"]): r for r in at_one}
    ratios = []
    for row in nose:
        if row["index"] == 1.0 or not stepped(row):
            continue
        base = reference[(row["kind"], row["speed"])]
        for key in ("onset_90_seconds", "slip_settle_seconds_after_switch"):
            if row[key] is not None and base[key]:
                ratios.append(
                    {
                        "index": row["index"],
                        "kind": row["kind"],
                        "speed": row["speed"],
                        "metric": key,
                        "ratio_over_index": _r(row[key] / base[key] / row["index"]),
                    }
                )
    tick_slack = 1.5 * config.dt / min(r["onset_90_seconds"] or 1.0 for r in at_one)
    checks["nose_times_scale_with_index"] = {
        "pass": all(abs(r["ratio_over_index"] - 1.0) <= tick_slack + 0.05 for r in ratios),
        "ratios": ratios,
    }

    free = {(r["power_mode"], r["turn"]): r for r in artifact["terminal_field_free"]}
    worst = 0.0
    for row in artifact["terminal"]:
        if row["index"] != 1.0:
            continue
        other = free[(row["power_mode"], row["turn"])]
        worst = max(worst, abs(other["proper_speed"] / row["proper_speed"] - 1.0))
    checks["field_free_matches_field_path"] = {
        "pass": worst <= 1e-3,
        "worst_relative_error": _r(worst, 6),
    }
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--play-envs", type=int, default=64)
    parser.add_argument("--play-decisions", type=int, default=1500)
    parser.add_argument("--play-seeds", default="271828,314159")
    parser.add_argument("--skip-play", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    live = frontline_ship_config(SHIP_CONFIG)

    started = time.perf_counter()
    with torch.inference_mode():
        artifact = {
            "schema": "boost-and-broadside-flight-envelope-v2",
            "git_commit": _git_head(),
            "units": {
                "speed": "proper px/s unless the key says world",
                "angle": "degrees",
                "ship_length_px": SHIP_LENGTH_PX,
            },
            "live_config": {
                "description": "frontline_ship_config(SHIP_CONFIG), field path",
                **asdict(live),
            },
            "terminal": measure_terminal(live, device),
            "terminal_field_free": measure_terminal(live, device, path="ambient", indices=(1.0,)),
            "em_grid": measure_em_grid(live, device),
            "straight_line": measure_straight_line(live, device),
            "turn_transients": measure_turn_transients(live, device, indices=(0.5, 1.0, 2.0)),
            "nose": measure_nose(live, device, indices=(0.5, 1.0, 2.0)),
            "bullets": measure_bullets(live, device),
        }
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    artifact["validation"] = validate(artifact, baseline, live)
    if not args.skip_play:
        seeds = tuple(int(s) for s in args.play_seeds.split(",") if s)
        artifact["play"] = measure_play(
            device, envs=args.play_envs, decisions=args.play_decisions, seeds=seeds
        )
    artifact["runtime"] = {
        "device": str(device),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=1) + "\n")
    for name, check in artifact["validation"].items():
        verdict = check.get("pass")
        print(f"{name:34s} {'-' if verdict is None else ('PASS' if verdict else 'FAIL')}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
