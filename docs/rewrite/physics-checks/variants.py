"""Flight-update variants to locate the first-order error. Not production code."""
from __future__ import annotations
import os
import torch
from boost_and_broadside.env.physics import (
    _apply_thrust_impulse_with_power, _quadratic_drag_scale, _drift, _nose_step,
    slip_command, slip_lift_drag, EPS,
)

FLAGS = set(filter(None, os.environ.get("VARIANT", "").split(",")))


def _thrust(state, thrust_mag, d, config):
    return _apply_thrust_impulse_with_power(state, thrust_mag, d, config)


def _drag(state, drag_coeff, d):
    speed = state.ship_vel.abs()
    direction = state.ship_vel / speed.clamp(min=EPS)
    state.ship_vel = direction * speed * _quadratic_drag_scale(state.ship_vel, drag_coeff, d)
    return state


def _lift(state, lift_coeff, d, slip_rotation):
    speed = state.ship_vel.abs()
    rot = torch.polar(torch.ones_like(speed), lift_coeff * speed * d)
    state.ship_vel = state.ship_vel * rot
    if "track" in FLAGS:  # thrust direction follows the rotating path
        state.ship_attitude = state.ship_vel / state.ship_vel.abs().clamp(min=EPS) * slip_rotation
    return state


def update_kinematics(state, actions, config, tables):
    thrust_mag = tables.thrust[actions[..., 0].long()]
    turn_side = tables.turn_side[actions[..., 1].long()]
    turn_sharp = tables.turn_sharp[actions[..., 1].long()]
    index = state.ship_local_index
    proper_speed = index * state.ship_vel.abs()
    below = proper_speed < config.min_speed
    thrust_mag = torch.where(below & (thrust_mag < 0), torch.zeros_like(thrust_mag), thrust_mag)
    prev_att = state.ship_attitude
    vdir = state.ship_vel / state.ship_vel.abs().clamp(min=EPS)
    command = slip_command(turn_side, turn_sharp, proper_speed, config)
    if "nonose" in FLAGS:  # instant nose: slip = command
        flown, rate = command, torch.zeros_like(command)
    else:
        flown, rate = _nose_step(state.ship_slip, state.ship_slip_rate, command, index, config)
    held = torch.angle(prev_att * torch.conj(vdir))
    slip = torch.where(below, held, flown)
    state.ship_slip_rate = torch.where(below, torch.zeros_like(rate), rate)
    srot = torch.polar(torch.ones_like(slip), slip)
    state.ship_attitude = torch.where(below, prev_att, vdir * srot)
    lift_slip = slip
    if "midslip" in FLAGS:
        lift_slip = torch.where(below, slip, 0.5 * (state.ship_slip + slip))
    lift_c, drag_c = slip_lift_drag(lift_slip, config)
    lift_c = torch.where(below, torch.zeros_like(lift_c), lift_c)
    h = 0.5 * config.dt
    state = _thrust(state, thrust_mag, h, config)
    state = _drag(state, drag_c, h)
    state = _lift(state, lift_c, h, srot)
    state = _drift(state, config)
    if "mirror" in FLAGS:  # Strang: second half in reverse order
        state = _lift(state, lift_c, h, srot)
        state = _drag(state, drag_c, h)
        state = _thrust(state, thrust_mag, h, config)
    else:
        state = _thrust(state, thrust_mag, h, config)
        state = _drag(state, drag_c, h)
        state = _lift(state, lift_c, h, srot)
    state.ship_power = torch.clamp(state.ship_power + config.passive_power_gain * config.dt, 0.0, config.max_power)
    sp = state.ship_vel.abs()
    state.ship_vel = torch.where(sp < EPS, EPS * state.ship_attitude, state.ship_vel)
    end_dir = state.ship_vel / state.ship_vel.abs().clamp(min=EPS)
    state.ship_attitude = torch.where(below, state.ship_attitude, end_dir * srot)
    end_held = torch.angle(state.ship_attitude * torch.conj(end_dir))
    state.ship_slip = torch.where(below, end_held, slip)
    state.ship_ang_vel = torch.angle(state.ship_attitude * torch.conj(prev_att)) / config.dt
    return state


import dataclasses

def _forces_substepped(state, thrust_mag_raw, turn_side, turn_sharp, config, duration, K):
    """Nose + thrust + drag + lift over `duration` in K symmetric substeps (no transport)."""
    sub = duration / K
    cfg = dataclasses.replace(config, dt=sub)
    for _ in range(K):
        index = state.ship_local_index
        proper_speed = index * state.ship_vel.abs()
        below = proper_speed < config.min_speed
        thrust_mag = torch.where(below & (thrust_mag_raw < 0), torch.zeros_like(thrust_mag_raw), thrust_mag_raw)
        prev_att = state.ship_attitude
        vdir = state.ship_vel / state.ship_vel.abs().clamp(min=EPS)
        command = slip_command(turn_side, turn_sharp, proper_speed, config)
        flown, rate = _nose_step(state.ship_slip, state.ship_slip_rate, command, index, cfg)
        held = torch.angle(prev_att * torch.conj(vdir))
        slip = torch.where(below, held, flown)
        state.ship_slip_rate = torch.where(below, torch.zeros_like(rate), rate)
        srot = torch.polar(torch.ones_like(slip), slip)
        state.ship_attitude = torch.where(below, prev_att, vdir * srot)
        lift_c, drag_c = slip_lift_drag(slip, config)
        lift_c = torch.where(below, torch.zeros_like(lift_c), lift_c)
        h = 0.5 * sub
        state = _thrust(state, thrust_mag, h, config)
        state = _drag(state, drag_c, h)
        state = _lift(state, lift_c, sub, srot)
        state = _drag(state, drag_c, h)
        state = _thrust(state, thrust_mag, h, config)
        end_dir = state.ship_vel / state.ship_vel.abs().clamp(min=EPS)
        state.ship_attitude = torch.where(below, state.ship_attitude, end_dir * srot)
        state.ship_slip = torch.where(below, torch.angle(state.ship_attitude * torch.conj(end_dir)), slip)
    return state


def update_kinematics_sub(state, actions, config, tables):
    K = int(os.environ.get("K", "4"))
    thrust_mag = tables.thrust[actions[..., 0].long()]
    turn_side = tables.turn_side[actions[..., 1].long()]
    turn_sharp = tables.turn_sharp[actions[..., 1].long()]
    prev_att = state.ship_attitude
    state = _forces_substepped(state, thrust_mag, turn_side, turn_sharp, config, 0.5 * config.dt, K)
    state = _drift(state, config)
    state = _forces_substepped(state, thrust_mag, turn_side, turn_sharp, config, 0.5 * config.dt, K)
    state.ship_power = torch.clamp(state.ship_power + config.passive_power_gain * config.dt, 0.0, config.max_power)
    state.ship_ang_vel = torch.angle(state.ship_attitude * torch.conj(prev_att)) / config.dt
    return state
