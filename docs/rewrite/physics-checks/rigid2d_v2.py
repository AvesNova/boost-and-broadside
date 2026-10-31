"""2D rigid-body ship with refractive fields, in x/y and in 2D PGA (clean, body-frame). JAX.

Physics in the map-local frame, no wrapping. Fields: a refractive index n(x) makes space
conformally curved (metric n^2 I). The ship flies today's flight model in proper time
(d tau = dt / n) with proper velocity w = n v; the field adds only geodesic bending of w
at rate omega_f = |v| (v_hat x grad ln n), which conserves |w| (generalized energy).

Per step (both versions, identical math): half kick (RK2 on body twist r, w_b and power,
pose held) | exact screw of the pose with body velocity w_b / n | field evaluation at the
new position (once per step) | half kick.
"""
import math
from dataclasses import dataclass
import jax, jax.numpy as jnp


@dataclass(frozen=True)
class Ship:
    cl_max: float = 0.027
    alpha_s: float = math.radians(15.0)
    cd0: float = 8e-4
    k_ind: float = (5e-3 - 8e-4) / 0.027 ** 2
    a_max: float = 270.0
    w_c: float = 12.3
    zeta: float = 0.7
    w_floor: float = 1.0
    max_power: float = 100.0
    psc: float = 200.0
    regen: float = 10.0
    band: float = 0.02
    eps: float = 1e-6

    @property
    def u_corner(self):
        return math.sqrt(self.a_max / self.cl_max)


THRUST = jnp.array([8.0, 80.0, -80.0])
REV_LO, REV_HI = 10.0, 30.0   # reverse thrust fades out below REV_HI, zero at REV_LO (smoothstep)


def eff_thrust(sh, T, power, u):
    band = sh.band * sh.max_power
    z = jnp.clip((u - REV_LO) / (REV_HI - REV_LO), 0.0, 1.0)
    rev = jnp.clip((sh.max_power - power) / band, 0.0, 1.0) * z * z * (3.0 - 2.0 * z)
    return T * jnp.where(T > 0, jnp.clip(power / band, 0.0, 1.0), rev)
STICK = jnp.array([0.0, -1 / 3, 1 / 3, -2 / 3, 2 / 3, -1.0, 1.0])


# ------------------------------------------------------------------ fields
def fields_eval(fx, fy, fr, fwid, flog, x, y):
    """ln n = sum_k L_k s_k(x) with a quintic smoothstep band (simplified: no union rule).
    Returns n and grad ln n. fx..flog: (F,); x, y: (S,)."""
    dx, dy = x[:, None] - fx[None], y[:, None] - fy[None]
    d = jnp.sqrt(dx * dx + dy * dy)
    zu = 0.5 - (d - fr[None]) / fwid[None]
    z = jnp.clip(zu, 0.0, 1.0)
    prof = z * z * z * (z * (z * 6.0 - 15.0) + 10.0)
    dprof = 30.0 * z * z * (z - 1.0) ** 2
    fac = jnp.where((zu > 0) & (zu < 1) & (d > 1e-6), -dprof / fwid[None] / jnp.maximum(d, 1e-6), 0.0) * flog[None]
    logn = (prof * flog[None]).sum(1)
    return jnp.exp(logn), (fac * dx).sum(1), (fac * dy).sum(1)


# ------------------------------------------------------------------ flight (body frame, proper units)
def lift(sh, alpha):
    x = jnp.clip(alpha / sh.alpha_s, -2.0, 2.0)
    return sh.cl_max * (2.0 * x - x * jnp.abs(x))


def alpha_cmd(sh, stick, u):
    a_w = jnp.abs(stick) * sh.cl_max * u * u
    x2 = (a_w / sh.a_max) ** 2
    a = a_w / jnp.sqrt(jnp.sqrt(1.0 + x2 * x2))
    c = jnp.clip(a / jnp.maximum(u * u, sh.eps) / sh.cl_max, 0.0, 1.0)
    return jnp.sign(stick) * sh.alpha_s * (1.0 - jnp.sqrt(1.0 - c))


def twist_rates(sh, r, wx, wy, power, n, gbx, gby, T, stick):
    """d/dt of (r, w_b, power) in world time. w_b = body-frame proper velocity; g_b = body
    gradient of ln n. Nose at body +x; alpha = angle from velocity to nose."""
    u = jnp.sqrt(wx * wx + wy * wy)
    inv = 1.0 / jnp.maximum(u, sh.eps)
    dx, dy = wx * inv, wy * inv
    alpha = jnp.arctan2(-dy, dx)
    cl = lift(sh, alpha)
    cd = sh.cd0 + sh.k_ind * cl * cl
    Te = eff_thrust(sh, T, power, u)
    # proper-time accelerations in the body frame: thrust on +x, lift perpendicular to w, drag along -w
    fx = Te + (-cl * u * u * dy) - cd * u * u * dx
    fy = cl * u * u * dx - cd * u * u * dy
    # geodesic bending (world time): rotation of w at omega_f = |v| (v_hat x g) = (u / n) (d x g)
    omega_f = (u / n) * (dx * gby - dy * gbx)
    # world turn rate of the flight path (world time): flight forces (w x f / |w|^2, proper) / n + bending
    theta_dot = (wx * fy - wy * fx) * inv * inv / n + omega_f
    w = jnp.maximum(sh.w_c * u / sh.u_corner, sh.w_floor)
    alpha_dot = r - theta_dot     # alpha = psi - theta
    rdot = (w * w * (alpha_cmd(sh, stick, u) - alpha) - 2.0 * sh.zeta * w * n * alpha_dot) / (n * n)
    k = omega_f - r                                   # rotation of w_b in the body frame
    dwx = fx / n - k * wy
    dwy = fy / n + k * wx
    dp = sh.regen - Te * wx / sh.psc / n             # thrust work in proper units, regen in world time
    return rdot, dwx, dwy, dp


import os
MIDN = bool(int(os.environ.get("MIDN", "1")))


def kick(sh, r, wx, wy, power, n, gbx, gby, T, stick, h):
    if EXPK:
        return kick_exp(sh, r, wx, wy, power, n, gbx, gby, T, stick, h)
    k = twist_rates(sh, r, wx, wy, power, n, gbx, gby, T, stick)
    m = (r + 0.5 * h * k[0], wx + 0.5 * h * k[1], wy + 0.5 * h * k[2], power + 0.5 * h * k[3])
    k = twist_rates(sh, *m, n, gbx, gby, T, stick)
    return r + h * k[0], wx + h * k[1], wy + h * k[2], jnp.clip(power + h * k[3], 0.0, sh.max_power)


EXPK = bool(int(os.environ.get("EXPK", "1")))


def lift_slope(sh, alpha):
    x = jnp.clip(alpha / sh.alpha_s, -2.0, 2.0)
    return jnp.where(jnp.abs(alpha / sh.alpha_s) < 2.0, sh.cl_max / sh.alpha_s * (2.0 - 2.0 * jnp.abs(x)), 0.0)


def kick_exp(sh, r, wx, wy, power, n, gbx, gby, T, stick, h):
    """Exponential kick. The fast (alpha, r) oscillator is solved exactly with coefficients
    taken at the half-kick's predicted midpoint speed (linearized lift slope); the velocity
    direction is then set exactly (|w| untouched: lift and bending do no work); speed and
    power by RK2."""
    u0 = jnp.sqrt(wx * wx + wy * wy)
    us0 = jnp.maximum(u0, sh.eps)
    alpha0 = jnp.arctan2(-wy, wx)

    def speed_rates(uu, pp, aa):
        cl = lift(sh, aa); cd = sh.cd0 + sh.k_ind * cl * cl
        te = eff_thrust(sh, T, pp, uu)
        return (te * jnp.cos(aa) - cd * uu * uu) / n, sh.regen - te * uu * jnp.cos(aa) / sh.psc / n

    du0, dp0 = speed_rates(u0, power, alpha0)
    u = jnp.maximum(u0 + 0.5 * h * du0, sh.eps)            # predicted midpoint speed
    pm = power + 0.5 * h * dp0
    Te = eff_thrust(sh, T, pm, u)
    dx, dy = wx / us0, wy / us0
    omega_f = (u / n) * (dx * gby - dy * gbx)
    theta_dot0 = (lift(sh, alpha0) * u * u + Te * jnp.sin(alpha0)) / (u * n) + omega_f
    kappa = (lift_slope(sh, alpha0) * u * u + Te * jnp.cos(alpha0)) / (u * n)
    Om = jnp.maximum(sh.w_c * u / sh.u_corner, sh.w_floor) / n
    d_alpha = alpha_cmd(sh, stick, u) - alpha0
    a11, a12, a21, a22 = -kappa, 1.0, -Om * Om + 2.0 * sh.zeta * Om * kappa, -2.0 * sh.zeta * Om
    e_s, z_s = d_alpha, kappa * d_alpha
    y1, y2 = -e_s, (r - theta_dot0) - z_s
    m = 0.5 * (a11 + a22)
    disc = m * m - Om * Om
    q = jnp.sqrt(jnp.abs(disc))
    qh = q * h
    osc = disc < 0
    C = jnp.where(osc, jnp.cos(qh), jnp.cosh(jnp.minimum(qh, 30.0)))
    Sq = jnp.where(qh > 1e-6, jnp.where(osc, jnp.sin(qh), jnp.sinh(jnp.minimum(qh, 30.0))) / jnp.maximum(q, 1e-12), h)
    em = jnp.exp(m * h)
    n1 = em * (C * y1 + Sq * ((a11 - m) * y1 + a12 * y2))
    n2 = em * (C * y2 + Sq * (a21 * y1 + (a22 - m) * y2))
    alpha1 = alpha0 + e_s + n1
    r1 = theta_dot0 + z_s + n2
    am = 0.5 * (alpha0 + alpha1)
    du, dp = speed_rates(u0, power, am)
    du, dp = speed_rates(jnp.maximum(u0 + 0.5 * h * du, sh.eps), power + 0.5 * h * dp, am)
    u1 = jnp.maximum(u0 + h * du, sh.eps)
    return r1, u1 * jnp.cos(alpha1), -u1 * jnp.sin(alpha1), jnp.clip(power + h * dp, 0.0, sh.max_power)


def drift_index(n, gbx, gby, wx, wy, h):
    """n at the step midpoint from ln n + 1/2 g . displacement (free: g is already known)."""
    if not MIDN:
        return n
    return n * jnp.exp(0.5 * h * (gbx * wx + gby * wy) / n)


# ------------------------------------------------------------------ x/y version
def xy_step(sh, field, st, act, h):
    x, y, c, s, r, wx, wy, power, n, gx, gy = st
    T, stick = THRUST[act[..., 0]].astype(x.dtype), STICK[act[..., 1]].astype(x.dtype)
    gbx, gby = c * gx + s * gy, -s * gx + c * gy
    r, wx, wy, power = kick(sh, r, wx, wy, power, n, gbx, gby, T, stick, 0.5 * h)
    # exact screw: rotate by phi = r h, body displacement = v_b (e^{i phi} - 1) / (i phi) h
    phi = r * h
    cp, sp = jnp.cos(phi), jnp.sin(phi)
    small = jnp.abs(phi) < 1e-4
    a1 = jnp.where(small, 1.0 - phi * phi / 6.0, sp / jnp.where(small, 1.0, phi)) * h
    a2 = jnp.where(small, 0.5 * phi, (1.0 - cp) / jnp.where(small, 1.0, phi)) * h
    nm = drift_index(n, gbx, gby, wx, wy, h)
    vx, vy = wx / nm, wy / nm
    bx, by = vx * a1 - vy * a2, vx * a2 + vy * a1
    x, y = x + c * bx - s * by, y + s * bx + c * by
    c, s = c * cp - s * sp, s * cp + c * sp
    n2, gx, gy = fields_eval(*field, x, y)
    wx, wy = wx, wy  # proper velocity is continuous across the field change
    gbx, gby = c * gx + s * gy, -s * gx + c * gy
    r, wx, wy, power = kick(sh, r, wx, wy, power, n2, gbx, gby, T, stick, 0.5 * h)
    return x, y, c, s, r, wx, wy, power, n2, gx, gy


# ------------------------------------------------------------------ PGA version
# Motor M = s0 + p e12 + a e01 + b e02. Rotor(psi) = cos(psi/2) - sin(psi/2) e12;
# translator(t) = 1 - (tx/2) e01 - (ty/2) e02; M X M~ maps body to world.
def motor_mul(m1, m2):
    s, p, a, b = m1
    t, q, c, d = m2
    return (s * t - p * q, p * t + q * s, a * t - b * q + c * s + d * p, a * q + b * t - c * p + d * s)


def motor_exp(r, vx, vy, h):
    """exp(-1/2 (r e12 + vx e01 + vy e02) h) = cos(rh/2) + sinc(rh/2) B (2D PGA: B^2 = -(rh/2)^2)."""
    half = 0.5 * r * h
    small = jnp.abs(half) < 1e-4
    sinc = jnp.where(small, 1.0 - half * half / 6.0, jnp.sin(half) / jnp.where(small, 1.0, half))
    return jnp.cos(half), -sinc * half, -0.5 * sinc * vx * h, -0.5 * sinc * vy * h


def motor_position(m):
    s, p, a, b = m
    return -2.0 * (a * s + b * p), 2.0 * (a * p - b * s)


def motor_heading(m):
    s, p, _, _ = m
    return s * s - p * p, -2.0 * p * s


def pga_step(sh, field, st, act, h):
    m, r, wx, wy, power, n, gx, gy = st
    T, stick = THRUST[act[..., 0]].astype(power.dtype), STICK[act[..., 1]].astype(power.dtype)
    c, s = motor_heading(m)
    gbx, gby = c * gx + s * gy, -s * gx + c * gy
    r, wx, wy, power = kick(sh, r, wx, wy, power, n, gbx, gby, T, stick, 0.5 * h)
    nm = drift_index(n, gbx, gby, wx, wy, h)
    m = motor_mul(m, motor_exp(r, wx / nm, wy / nm, h))
    x, y = motor_position(m)
    n2, gx, gy = fields_eval(*field, x, y)
    c, s = motor_heading(m)
    gbx, gby = c * gx + s * gy, -s * gx + c * gy
    r, wx, wy, power = kick(sh, r, wx, wy, power, n2, gbx, gby, T, stick, 0.5 * h)
    return m, r, wx, wy, power, n2, gx, gy
