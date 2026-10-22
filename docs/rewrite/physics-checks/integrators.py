"""Candidate flight integrators for the physics-first comparison. Not production code.

C0  today's `physics._update_kinematics`
C1  symmetric Strang split, nose stepped per half: F(dt/2) | transport(dt) | F(dt/2), each F
    = nose, thrust/2, drag/2, lift, drag/2, thrust/2 (K substeps per half)
C3  exact flow: field rotation kick(dt/2) | exact thrust+drag+lift+motion over dt in K slip
    pieces (closed-form speed, Gauss quadrature for heading and position, power as an event,
    regen inside the flow) | rescale to proper speed | field rotation kick(dt/2)
"""
from __future__ import annotations
import dataclasses, math
import torch
from boost_and_broadside.env import physics
from boost_and_broadside.env.physics import (
    _drift, _transport_through_fields, _nose_step, _evaluate_state_fields, _wrap_positions,
    slip_command, slip_lift_drag, EPS,
)
import os
import variants
MIDSLIP = bool(os.environ.get('MIDSLIP'))

G2 = ((0.5 - 0.5 / math.sqrt(3), 0.5), (0.5 + 0.5 / math.sqrt(3), 0.5))
G3 = ((0.5 - 0.5 * math.sqrt(0.6), 5 / 18), (0.5, 8 / 18), (0.5 + 0.5 * math.sqrt(0.6), 5 / 18))


def _decode(actions, tables):
    return (tables.thrust[actions[..., 0].long()], tables.turn_side[actions[..., 1].long()],
            tables.turn_sharp[actions[..., 1].long()])


def c0(state, actions, config, tables, K=1):
    return physics._update_kinematics(state, actions, config, tables)


def _transport(state, config):
    return _drift(state, config) if state.num_fields == 0 else _transport_through_fields(state, config)


def c1(state, actions, config, tables, K=1):
    thrust, side, sharp = _decode(actions, tables)
    prev = state.ship_attitude
    state = variants._forces_substepped(state, thrust, side, sharp, config, 0.5 * config.dt, K)
    state = _transport(state, config)
    state = variants._forces_substepped(state, thrust, side, sharp, config, 0.5 * config.dt, K)
    state.ship_power = torch.clamp(state.ship_power + config.passive_power_gain * config.dt, 0.0, config.max_power)
    state.ship_ang_vel = torch.angle(state.ship_attitude * torch.conj(prev)) / config.dt
    return state


# ---------------------------------------------------------------- exact flow pieces
def _speed_and_S(s0, A, c, t):
    """Speed and integral of speed for ds/dt = A - c s^2 (c > 0), elementwise."""
    pos, neg = A > 1e-6, A < -1e-6
    k = torch.sqrt(A.abs() / c).clamp(min=1e-6)
    r = torch.sqrt(A.abs() * c)
    x = r * t
    th, tn = torch.tanh(x), torch.tan(torch.clamp(x, max=1.5))
    s_pos = k * (s0 + k * th) / (k + s0 * th)
    s_neg = k * (s0 - k * tn) / (k + s0 * tn)
    s_zero = s0 / (1 + c * s0 * t)
    # S = ln(u)/c, u = cosh x + (s0/k) sinh x  (pos) or cos x + (s0/k) sin x (neg); log1p for accuracy
    u_pos_m1 = 2 * torch.sinh(0.5 * x) ** 2 + (s0 / k) * torch.sinh(x)
    u_neg_m1 = -2 * torch.sin(0.5 * x) ** 2 + (s0 / k) * torch.sin(x)
    S_pos, S_neg = torch.log1p(u_pos_m1) / c, torch.log1p(u_neg_m1.clamp(min=-0.999999)) / c
    S_zero = torch.log1p(c * s0 * t) / c
    s = torch.where(pos, s_pos, torch.where(neg, s_neg, s_zero))
    S = torch.where(pos, S_pos, torch.where(neg, S_neg, S_zero))
    return s.clamp(min=1e-3), S


def _segment(pos, s0, theta0, A, lat, CL, c, h):
    """Exact speed; heading and position by Gauss quadrature, over duration h (tensor)."""
    def theta_at(t):
        _, S = _speed_and_S(s0, A, c, t)
        Q = sum(w * t / _speed_and_S(s0, A, c, x * t)[0] for x, w in G2)
        return theta0 + CL * S + lat * Q
    disp = 0
    for x, w in G3:
        t = x * h
        s_t, _ = _speed_and_S(s0, A, c, t)
        disp = disp + w * h * s_t * torch.polar(torch.ones_like(s_t), theta_at(t))
    s_h, _ = _speed_and_S(s0, A, c, h)
    return pos + disp, s_h, theta_at(h)


def _exact_piece(state, thrust_raw, side, sharp, config, h):
    index = state.ship_local_index
    mass = index.square()
    speed = state.ship_vel.abs()
    below = index * speed < config.min_speed
    vdir = state.ship_vel / speed.clamp(min=EPS)
    command = slip_command(side, sharp, index * speed, config)
    cfg = dataclasses.replace(config, dt=h)
    if MIDSLIP:  # slip at the piece midpoint drives the forces (midpoint rule)
        half = dataclasses.replace(config, dt=0.5 * h)
        mid_slip, mid_rate = _nose_step(state.ship_slip, state.ship_slip_rate, command, index, half)
        flown, rate = _nose_step(mid_slip, mid_rate, command, index, half)
        force_slip = mid_slip
    else:
        flown, rate = _nose_step(state.ship_slip, state.ship_slip_rate, command, index, cfg)
        force_slip = flown
    slip, srot = flown, torch.polar(torch.ones_like(flown), flown)
    lift_c, drag_c = slip_lift_drag(force_slip, config)
    psc, regen = config.power_speed_constant, config.passive_power_gain

    T = torch.where((thrust_raw > 0) & (state.ship_power <= 0), torch.zeros_like(thrust_raw), thrust_raw)
    cos_s, sin_s = torch.cos(force_slip), torch.sin(force_slip)
    H = torch.full_like(speed, h)

    def used(t, Tm):  # power spent minus regen over [0, t]
        _, S = _speed_and_S(speed, Tm / mass * cos_s, drag_c, t)
        return Tm * cos_s * S / psc - regen * t

    # event 1: forward thrust empties the power store
    lo, hi = torch.zeros_like(speed), H.clone()
    empties = (T > 0) & (used(H, T) > state.ship_power)
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        over = used(mid, T) > state.ship_power
        hi, lo = torch.where(over, mid, hi), torch.where(over, lo, mid)
    t_cut = torch.where(empties, lo, H)
    # event 1b: reverse thrust fills the power store (recovery stops; the original caps it too)
    lo, hi = torch.zeros_like(speed), H.clone()
    fills = (T < 0) & (state.ship_power - used(H, T) > config.max_power)
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        over = state.ship_power - used(mid, T) > config.max_power
        hi, lo = torch.where(over, mid, hi), torch.where(over, lo, mid)
    t_cut = torch.where(fills, torch.minimum(lo, t_cut), t_cut)
    # event 2: reverse thrust reaches the stall floor
    A_rev = T / mass * cos_s
    k = torch.sqrt(A_rev.abs() / drag_c)
    s_min = torch.full_like(speed, 2 * config.min_speed)
    tau = k * (speed - s_min) / (k * k + speed * s_min)
    t_stop = torch.atan(tau.clamp(min=0)) / torch.sqrt(A_rev.abs() * drag_c).clamp(min=1e-9)
    t_cut = torch.where((A_rev < 0) & (t_stop < t_cut), t_stop, t_cut)

    theta0 = torch.angle(vdir)
    pos1, s1, th1 = _segment(state.ship_pos, speed, theta0, T / mass * cos_s, T / mass * sin_s, lift_c, drag_c, t_cut)
    p1 = state.ship_power - used(t_cut, T)
    # after the event: throttle to what regen sustains (power empty) or coast (reverse stop)
    T2 = torch.where(empties, torch.minimum(T, regen * psc / (s1 * cos_s).clamp(min=1e-3)), torch.zeros_like(T))
    T2 = torch.where(empties, T2, torch.zeros_like(T))
    rest = H - t_cut
    pos2, s2, th2 = _segment(pos1, s1, th1, T2 / mass * cos_s, T2 / mass * sin_s, lift_c, drag_c, rest)
    _, S2 = _speed_and_S(s1, T2 / mass * cos_s, drag_c, rest)
    p2 = p1 - (T2 * cos_s * S2 / psc - regen * rest)

    new_vel = s2 * torch.polar(torch.ones_like(s2), th2)
    state.ship_pos = _wrap_positions(pos2, config.world_size)
    state.ship_vel = new_vel
    state.ship_power = p2.clamp(0.0, config.max_power)
    state.ship_slip, state.ship_slip_rate = slip, rate
    state.ship_attitude = new_vel / new_vel.abs().clamp(min=EPS) * srot
    return state, below


def _field_kick(state, h):
    if state.num_fields == 0:
        return state
    g = state.ship_field_gradient / state.ship_local_index.clamp(min=EPS)
    speed = state.ship_vel.abs()
    vdir = state.ship_vel / speed.clamp(min=EPS)
    omega = speed * (torch.conj(vdir) * g).imag
    state.ship_vel = state.ship_vel * torch.polar(torch.ones_like(omega), omega * h)
    return state


def c3(state, actions, config, tables, K=1):
    thrust, side, sharp = _decode(actions, tables)
    prev = state.ship_attitude
    state = _field_kick(state, 0.5 * config.dt)
    snap = dataclasses.replace(state) if False else None
    for _ in range(K):
        before = {f: getattr(state, f).clone() for f in ("ship_pos", "ship_vel", "ship_attitude", "ship_slip", "ship_slip_rate", "ship_power")}
        state, below = _exact_piece(state, thrust, side, sharp, config, config.dt / K)
        if bool(below.any()):  # rare stall: redo those ships with the split scheme
            fallback = {f: getattr(state, f).clone() for f in before}
            for f, v in before.items(): setattr(state, f, v)
            sub = dataclasses.replace(config, dt=config.dt / K)
            state = c1(state, actions, sub, tables, 1)
            for f in before:
                setattr(state, f, torch.where(below, getattr(state, f), fallback[f]))
    if state.num_fields:
        n_old = state.ship_local_index
        ev = _evaluate_state_fields(state.ship_pos, state, config.world_size)
        state.ship_vel = state.ship_vel * (n_old / ev.index)
        state.ship_local_index, state.ship_field_gradient = ev.index, ev.grad_index
        state.ship_attitude = state.ship_vel / state.ship_vel.abs().clamp(min=EPS) * torch.polar(torch.ones_like(state.ship_slip), state.ship_slip)
    state = _field_kick(state, 0.5 * config.dt)
    state.ship_attitude = state.ship_vel / state.ship_vel.abs().clamp(min=EPS) * torch.polar(torch.ones_like(state.ship_slip), state.ship_slip)
    state.ship_ang_vel = torch.angle(state.ship_attitude * torch.conj(prev)) / config.dt
    return state
