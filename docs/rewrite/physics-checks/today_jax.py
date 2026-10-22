"""Today's flight step (physics._update_kinematics, no fields) ported to JAX for a cost comparison."""
import math, jax, jax.numpy as jnp
from dataclasses import dataclass


@dataclass(frozen=True)
class Cfg:
    dt: float = 1 / 30
    min_speed: float = 1.0
    max_lateral_accel: float = 270.0
    max_lift_coeff: float = 27e-3
    normal_lift_coeff: float = 15e-3
    stall_angle: float = math.radians(15.0)
    slip_fade_speed: float = 35.0
    zero_slip_drag_coeff: float = 8e-4
    quad: float = 2e-3
    quart: float = 2e-3
    nose_frequency: float = 25.0
    max_slip_rate: float = math.radians(150.0)
    power_speed_constant: float = 200.0
    max_power: float = 100.0
    passive_power_gain: float = 10.0


THRUST = jnp.array([8.0, 80.0, -80.0]); SIDE = jnp.array([0.0, -1, 1, -1, 1]); SHARP = jnp.array([0, 0, 0, 1, 1], bool)


def slip_command(c, side, sharp, ps):
    sq = jnp.maximum(ps * ps, 1e-6)
    lift = jnp.where(sharp, jnp.minimum(c.max_lateral_accel / sq, c.max_lift_coeff), jnp.minimum(0.5 * c.max_lateral_accel / sq, c.normal_lift_coeff))
    frac = 1.0 - jnp.sqrt(jnp.maximum(1.0 - lift / c.max_lift_coeff, 0.0))
    return side * c.stall_angle * frac * jnp.minimum(ps / c.slip_fade_speed, 1.0)


def lift_drag(c, slip):
    f = jnp.abs(slip) / c.stall_angle
    lift = jnp.sign(slip) * c.max_lift_coeff * (1.0 - (1.0 - f) ** 2)
    lift = jnp.where(lift * jnp.sign(slip) > 0, lift, 0.0)
    f2 = jnp.minimum(f, 2.0) ** 2
    return lift, c.zero_slip_drag_coeff + f2 * (c.quad + c.quart * f2)


def nose(c, slip, rate, cmd):
    w, R, dt = c.nose_frequency, c.max_slip_rate, c.dt
    err = slip - cmd
    decay = jnp.exp(-w * dt); drive = rate + w * err
    ne, nr = (err + drive * dt) * decay, (rate - w * drive * dt) * decay
    ne = err + jnp.clip(ne - err, -R * dt, R * dt)
    nr = jnp.clip(nr, -R, R)
    arr = ne * err <= 0
    ne, nr = jnp.where(arr, 0.0, ne), jnp.where(arr, 0.0, nr)
    ns = cmd + ne
    b = jnp.maximum(jnp.abs(slip), c.stall_angle)
    return jnp.clip(ns, -b, b), nr


def thrust_impulse(c, vel, att, power, T, dur):
    pos = T > 0
    T = jnp.where(pos & (power <= 0), 0.0, T)
    dv = T * att * dur
    a = (vel * jnp.conj(dv)).real
    b = 0.5 * jnp.abs(dv) ** 2
    bs = jnp.maximum(b, 1e-12)
    frac = jnp.where(T < 0, jnp.clip(-a / (2 * bs), 0, 1), 1.0)
    frac = jnp.where(T == 0, 0.0, frac)
    de = a * frac + b * frac ** 2
    sp = power * c.power_speed_constant
    root = (-a + jnp.sqrt(jnp.maximum(a * a + 4 * b * sp, 0))) / (2 * bs)
    frac = jnp.where(de > sp, jnp.minimum(frac, jnp.clip(root, 0, 1)), frac)
    rec = (c.max_power - power) * c.power_speed_constant
    rroot = (-a - jnp.sqrt(jnp.maximum(a * a - 4 * b * rec, 0))) / (2 * bs)
    frac = jnp.where(-de > rec, jnp.minimum(frac, jnp.clip(rroot, 0, 1)), frac)
    de = a * frac + b * frac ** 2
    return vel + dv * frac, jnp.clip(power - de / c.power_speed_constant, 0, c.max_power)


def half(c, vel, att, power, T, lc, dc):
    d = 0.5 * c.dt
    vel, power = thrust_impulse(c, vel, att, power, T, d)
    sp = jnp.abs(vel); dirn = vel / jnp.maximum(sp, 1e-6)
    sp = sp / (1 + dc * sp * d)
    return dirn * sp * jnp.exp(1j * (lc * sp * d)), power


def step(c, st, act):
    pos, vel, att, slip, rate, power = st
    T, side, sharp = THRUST[act[..., 0]], SIDE[act[..., 1]], SHARP[act[..., 1]]
    ps = jnp.abs(vel); below = ps < c.min_speed
    T = jnp.where(below & (T < 0), 0.0, T)
    vdir = vel / jnp.maximum(ps, 1e-6)
    fs, fr = nose(c, slip, rate, slip_command(c, side, sharp, ps))
    held = jnp.angle(att * jnp.conj(vdir))
    slip = jnp.where(below, held, fs); rate = jnp.where(below, 0.0, fr)
    rot = jnp.exp(1j * slip)
    att = jnp.where(below, att, vdir * rot)
    lc, dc = lift_drag(c, slip); lc = jnp.where(below, 0.0, lc)
    vel, power = half(c, vel, att, power, T, lc, dc)
    pos = pos + vel * c.dt
    vel, power = half(c, vel, att, power, T, lc, dc)
    power = jnp.clip(power + c.passive_power_gain * c.dt, 0, c.max_power)
    end = vel / jnp.maximum(jnp.abs(vel), 1e-6)
    att = jnp.where(below, att, end * rot)
    return pos, vel, att, slip, rate, power
