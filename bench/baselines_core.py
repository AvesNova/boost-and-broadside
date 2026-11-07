"""The three one-tick baselines of the baselines bench (docs/rewrite/sections/
baselines-bench-brief.md) and their multi-tick forecasts, in numpy (float64), on arrays of
any common shape.

A state is a dict of arrays: x, y (m), vx, vy (world velocity, m/s), heading (rad),
omega (nose rate, rad/s), attack (angle of attack, rad), energy and shield (fractions of
the type's maximum), wait (shield recharge wait, s), cooldown (gun cooldown, whole ticks),
log_index (ln n), and, per ship, rate (the shield recharge per s as a fraction of the
type's maximum).

- P, persistence: nothing changes.
- R, rule: `bnb.models.yemong.delta.DeltaCode.baseline` (the arc at omega, v rotated by
  omega dt, heading + omega dt, persistence for the nose rate, angle of attack, energy and
  ln n, the shield recharge and the countdowns).
- E, extrapolation: last tick's rate (`extrapolate`); R where t - 1 is missing.
"""

import numpy as np

TRACK_MIN_SPEED = 1.0  # m/s: below it the velocity frame falls back to the nose
STATE = (
    "x",
    "y",
    "vx",
    "vy",
    "heading",
    "omega",
    "attack",
    "energy",
    "shield",
    "wait",
    "cooldown",
    "log_index",
)


def wrap(angle):
    """An angle wrapped to [-pi, pi)."""
    return np.remainder(angle + np.pi, 2.0 * np.pi) - np.pi


def rotate(x, y, angle):
    """(x, y) rotated by angle."""
    c, s = np.cos(angle), np.sin(angle)
    return c * x - s * y, s * x + c * y


def arc_series(angle, dt):
    """The game's drift series (A1, A2) for a turn of `angle` over dt (flight.flight_tick)."""
    square = angle * angle
    along = (1.0 - square * (1.0 / 6.0 - square / 120.0)) * dt
    across = angle * (0.5 - square * (1.0 / 24.0 - square / 720.0)) * dt
    return along, across


def arc_exact(angle, dt):
    """(sin a / a, (1 - cos a) / a) dt: the chord of a steady turn by `angle` over dt, as
    (along, across) factors of the velocity at the start (series near 0)."""
    small = np.abs(angle) < 1e-3
    safe = np.where(small, 1.0, angle)
    square = angle * angle
    along = np.where(small, 1.0 - square / 6.0, np.sin(safe) / safe)
    across = np.where(small, angle * (0.5 - square / 24.0), (1.0 - np.cos(safe)) / safe)
    return along * dt, across * dt


def track_frame(vx, vy, heading):
    """(ux, uy, fell back): the unit velocity direction, the nose below TRACK_MIN_SPEED."""
    speed = np.hypot(vx, vy)
    slow = speed < TRACK_MIN_SPEED
    safe = np.where(slow, 1.0, speed)
    ux = np.where(slow, np.cos(heading), vx / safe)
    uy = np.where(slow, np.sin(heading), vy / safe)
    return ux, uy, slow


def to_frame(ex, ey, ux, uy):
    """(along, across) of (ex, ey) in the frame of the unit vector (ux, uy); across to the
    left."""
    return ex * ux + ey * uy, -ex * uy + ey * ux


def persistence(state):
    """P: the state unchanged."""
    return {name: state[name] for name in STATE}


def rule(state, dt):
    """R: `DeltaCode.baseline`, with the cooldown in whole ticks."""
    angle = state["omega"] * dt
    along, across = arc_series(angle, dt)
    vx, vy = state["vx"], state["vy"]
    rx, ry = rotate(vx, vy, angle)
    out = persistence(state)
    out.update(
        x=state["x"] + vx * along - vy * across,
        y=state["y"] + vx * across + vy * along,
        vx=rx,
        vy=ry,
        heading=wrap(state["heading"] + angle),
        shield=np.minimum(
            1.0, state["shield"] + state["rate"] * np.clip(dt - state["wait"], 0.0, dt)
        ),
        wait=np.maximum(state["wait"] - dt, 0.0),
        cooldown=np.maximum(state["cooldown"] - 1.0, 0.0),
    )
    return out


def acceleration(previous, state, dt):
    """(tangential, normal) acceleration from t - 1 to t in the velocity frame at t - 1
    (normal to the left), and that frame's turn to the frame at t (rad)."""
    ux0, uy0, _ = track_frame(previous["vx"], previous["vy"], previous["heading"])
    ux1, uy1, _ = track_frame(state["vx"], state["vy"], state["heading"])
    ax = (state["vx"] - previous["vx"]) / dt
    ay = (state["vy"] - previous["vy"]) / dt
    tangential, normal = to_frame(ax, ay, ux0, uy0)
    turn = np.arctan2(ux0 * uy1 - uy0 * ux1, ux0 * ux1 + uy0 * uy1)
    return tangential, normal, turn


def extrapolate(previous, state, dt, shot_last, damage_last, has_previous):
    """E: last tick's rate (the brief's table), R where has_previous is False.

    shot_last: the ship fired on tick t - 1 (energy and cooldown jumped); damage_last: it
    took damage on tick t - 1 (shield and wait jumped). Returns (E, E-arc position (x, y)).
    E's position is the literal x + v dt + a dt^2 / 2 with the transported acceleration;
    E-arc is x + Arc(phi) (v + du / 2 u) dt, phi and du last tick's path turn and speed
    change (exact on a steady turn and on constant acceleration along a line).
    """
    fallback = rule(state, dt)
    tangential, normal, turn = acceleration(previous, state, dt)
    ux, uy, _ = track_frame(state["vx"], state["vy"], state["heading"])
    ax = tangential * ux - normal * uy  # the acceleration carried into the frame at t
    ay = tangential * uy + normal * ux
    vx, vy = state["vx"], state["vy"]
    omega_change = state["omega"] - previous["omega"]
    out = dict(
        x=state["x"] + vx * dt + 0.5 * ax * dt * dt,
        y=state["y"] + vy * dt + 0.5 * ay * dt * dt,
        vx=vx + ax * dt,
        vy=vy + ay * dt,
        heading=wrap(state["heading"] + state["omega"] * dt + 0.5 * omega_change * dt),
        omega=state["omega"] + omega_change,
        attack=2.0 * state["attack"] - previous["attack"],
        log_index=2.0 * state["log_index"] - previous["log_index"],
    )
    energy = np.clip(2.0 * state["energy"] - previous["energy"], 0.0, 1.0)
    out["energy"] = np.where(shot_last, fallback["energy"], energy)
    reset = state["wait"] - np.maximum(previous["wait"] - dt, 0.0) > 0.5 * dt
    shield_event = damage_last | reset
    shield = np.clip(2.0 * state["shield"] - previous["shield"], 0.0, 1.0)
    out["shield"] = np.where(shield_event, fallback["shield"], shield)
    wait = np.maximum(2.0 * state["wait"] - previous["wait"], 0.0)
    out["wait"] = np.where(shield_event, fallback["wait"], wait)
    fired = state["cooldown"] - np.maximum(previous["cooldown"] - 1.0, 0.0) > 0.5
    cooldown = np.maximum(2.0 * state["cooldown"] - previous["cooldown"], 0.0)
    out["cooldown"] = np.where(shot_last | fired, fallback["cooldown"], cooldown)
    change = np.hypot(vx, vy) - np.hypot(previous["vx"], previous["vy"])  # last tick's du
    along, across = arc_exact(turn, dt)
    bx, by = vx + 0.5 * change * ux, vy + 0.5 * change * uy
    arc_x = state["x"] + bx * along - by * across
    arc_y = state["y"] + bx * across + by * along
    out = {name: np.where(has_previous, out[name], fallback[name]) for name in STATE}
    arc_x = np.where(has_previous, arc_x, fallback["x"])
    arc_y = np.where(has_previous, arc_y, fallback["y"])
    return out, (arc_x, arc_y)


def forecasts(previous, state, dt, has_previous, horizons):
    """{horizon: {forecast: (x, y, vx, vy)}} of the state at t, dead reckoned k ticks with
    no new information: CV (constant velocity), R iterated (constant nose rate: the arc and
    v rotated by omega dt each tick) and E iterated (last tick's tangential and normal
    acceleration held in the velocity frame; R where t - 1 is missing)."""
    tangential, normal, _ = acceleration(previous, state, dt)
    tangential = np.where(has_previous, tangential, 0.0)
    normal = np.where(has_previous, normal, 0.0)
    x, y, vx, vy = state["x"], state["y"], state["vx"], state["vy"]
    heading, omega = state["heading"], state["omega"]
    rx, ry, rvx, rvy = x, y, vx, vy  # R
    ex, ey, evx, evy = x, y, vx, vy  # E
    ux, uy, _ = track_frame(vx, vy, heading)
    angle = omega * dt
    along, across = arc_series(angle, dt)
    cos, sin = np.cos(angle), np.sin(angle)
    out = {}
    for k in range(1, max(horizons) + 1):
        rx, ry = rx + rvx * along - rvy * across, ry + rvx * across + rvy * along
        rvx, rvy = rvx * cos - rvy * sin, rvx * sin + rvy * cos
        # E: the frame of the current velocity (kept while slower than TRACK_MIN_SPEED)
        speed = np.hypot(evx, evy)
        moving = speed >= TRACK_MIN_SPEED
        safe = np.where(moving, speed, 1.0)
        ux, uy = np.where(moving, evx / safe, ux), np.where(moving, evy / safe, uy)
        ax = tangential * ux - normal * uy
        ay = tangential * uy + normal * ux
        rule_x, rule_y = ex + evx * along - evy * across, ey + evx * across + evy * along
        rule_vx, rule_vy = evx * cos - evy * sin, evx * sin + evy * cos
        ex = np.where(has_previous, ex + evx * dt + 0.5 * ax * dt * dt, rule_x)
        ey = np.where(has_previous, ey + evy * dt + 0.5 * ay * dt * dt, rule_y)
        evx = np.where(has_previous, evx + ax * dt, rule_vx)
        evy = np.where(has_previous, evy + ay * dt, rule_vy)
        if k in horizons:
            out[k] = dict(
                P=(x, y, vx, vy),
                CV=(x + vx * k * dt, y + vy * k * dt, vx, vy),
                R=(rx, ry, rvx, rvy),
                E=(ex, ey, evx, evy),
            )
    return out
