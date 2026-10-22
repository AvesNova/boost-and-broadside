"""Alternative nose models, stepped exactly or smoothly, inside the symmetric split.

NOSE=ramp : slip moves toward its command at most max_slip_rate (exact, with arrival)
NOSE=tanh : slip_rate = R tanh((command - slip) / (R tau)), smooth, RK2 per piece
Forces in each piece use the slip at the piece midpoint; the command is re-evaluated per
piece. Reference = same model at 3840 Hz.
"""
from __future__ import annotations
import dataclasses, os, torch
from boost_and_broadside.env.physics import _drift, slip_command, slip_lift_drag, EPS
import variants

NOSE = os.environ.get("NOSE", "ramp")
TAU = float(os.environ.get("TAU", "0.08"))


def _advance(slip, command, R, h):
    if NOSE == "ramp":
        err = command - slip
        return slip + torch.sign(err) * torch.minimum(err.abs(), R * h)
    f = lambda x: R * torch.tanh((command - x) / (R * TAU))
    mid = slip + 0.5 * h * f(slip)
    return slip + h * f(mid)


def _forces(state, thrust, side, sharp, config, duration, K):
    sub = duration / K
    for _ in range(K):
        index = state.ship_local_index
        speed = state.ship_vel.abs()
        R = config.max_slip_rate / index
        command = slip_command(side, sharp, index * speed, config)
        mid = _advance(state.ship_slip, command, R, 0.5 * sub)
        end = _advance(state.ship_slip, command, R, sub)
        lift_c, drag_c = slip_lift_drag(mid, config)
        mrot = torch.polar(torch.ones_like(mid), mid)
        vdir = state.ship_vel / speed.clamp(min=EPS)
        state.ship_attitude = vdir * mrot
        h = 0.5 * sub
        state = variants._thrust(state, thrust, h, config)
        state = variants._drag(state, drag_c, h)
        state = variants._lift(state, lift_c, sub, mrot)
        state.ship_attitude = state.ship_vel / state.ship_vel.abs().clamp(min=EPS) * mrot
        state = variants._drag(state, drag_c, h)
        state = variants._thrust(state, thrust, h, config)
        state.ship_slip = end
        state.ship_attitude = state.ship_vel / state.ship_vel.abs().clamp(min=EPS) * torch.polar(torch.ones_like(end), end)
    return state


def step(state, actions, config, tables, K=1):
    thrust = tables.thrust[actions[..., 0].long()]
    side = tables.turn_side[actions[..., 1].long()]
    sharp = tables.turn_sharp[actions[..., 1].long()]
    state = _forces(state, thrust, side, sharp, config, 0.5 * config.dt, K)
    state = _drift(state, config)
    state = _forces(state, thrust, side, sharp, config, 0.5 * config.dt, K)
    state.ship_power = torch.clamp(state.ship_power + config.passive_power_gain * config.dt, 0.0, config.max_power)
    return state


# ---------------------------------------------------------------- g-command design
RESERVE = float(os.environ.get("RESERVE", "0.02"))  # soft power band, fraction of max


def _q_advance(q, q_cmd, R, h):
    f = lambda x: R * torch.tanh((q_cmd - x) / (R * TAU))
    mid = q + 0.5 * h * f(q)
    return q + h * f(mid)


def _lateral(q, speed, config):
    """Smooth min of the commanded lateral accel and the stall limit C_Lmax u^2."""
    want = q * config.max_lateral_accel
    cap = config.max_lift_coeff * speed.square()
    return want / torch.sqrt(1.0 + (want / cap.clamp(min=EPS)).square())


def _slip_from_lateral(a, speed, config):
    cl = (a.abs() / speed.square().clamp(min=EPS) / config.max_lift_coeff).clamp(max=1.0)
    return torch.sign(a) * config.stall_angle * (1.0 - torch.sqrt(1.0 - cl))


def _soft_thrust(thrust, power, config):
    band = RESERVE * config.max_power
    fwd = (power / band).clamp(0.0, 1.0)
    rev = ((config.max_power - power) / band).clamp(0.0, 1.0)
    return thrust * torch.where(thrust > 0, fwd, rev)


def _g_forces(state, thrust_raw, side, sharp, config, duration, K):
    sub = duration / K
    q_cmd = side * torch.where(sharp, 1.0, 0.5)
    R = config.max_slip_rate / config.stall_angle  # full-scale q per second, same reversal time
    for _ in range(K):
        q_mid = _q_advance(state.ship_slip, q_cmd, R, 0.5 * sub)
        q_end = _q_advance(state.ship_slip, q_cmd, R, sub)
        h = 0.5 * sub
        for part in ("thrust", "drag", "lift", "drag", "thrust"):
            speed = state.ship_vel.abs()
            a_lat = _lateral(q_mid, speed, config)
            slip = _slip_from_lateral(a_lat, speed, config)
            vdir = state.ship_vel / speed.clamp(min=EPS)
            state.ship_attitude = vdir * torch.polar(torch.ones_like(slip), slip)
            if part == "thrust":
                state = variants._thrust(state, _soft_thrust(thrust_raw, state.ship_power, config), h, config)
                state.ship_power = torch.clamp(state.ship_power + config.passive_power_gain * h, 0.0, config.max_power)
            elif part == "drag":
                _, drag_c = slip_lift_drag(slip, config)
                state = variants._drag(state, drag_c, 0.5 * h)
            else:
                turn = a_lat / speed.clamp(min=EPS) * sub
                state.ship_vel = state.ship_vel * torch.polar(torch.ones_like(turn), turn)
        state.ship_slip = q_end
    speed = state.ship_vel.abs()
    slip = _slip_from_lateral(_lateral(state.ship_slip, speed, config), speed, config)
    state.ship_attitude = state.ship_vel / speed.clamp(min=EPS) * torch.polar(torch.ones_like(slip), slip)
    return state


def g_step(state, actions, config, tables, K=1):
    thrust = tables.thrust[actions[..., 0].long()]
    side = tables.turn_side[actions[..., 1].long()]
    sharp = tables.turn_sharp[actions[..., 1].long()]
    state = _g_forces(state, thrust, side, sharp, config, 0.5 * config.dt, K)
    state = _drift(state, config)
    state = _g_forces(state, thrust, side, sharp, config, 0.5 * config.dt, K)
    return state
