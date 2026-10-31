"""Rigid-body ship v3: thrust as diving/climbing, a shared lift budget, storage variants. JAX.

Boost and reverse stand in for pitching down / up in gravity, with the power store as
altitude. So by default thrust acts along the flight path (velocity), not the nose, and
reverse (climb) draws on the same lift budget as turning: near stall there is no lift
left, so no climb, just as a stalled plane cannot pull up.

Lift budget  L(u) = softmin(C_Lmax u^2, a_max)            (soft g limit, decided)
Turn         a_lat = C_L(alpha) u^2  (alpha from the rigid-body nose)
Climb        a_climb = softmin(|T_rev|, L sqrt(1 - (a_lat / L)^2))   (turn has priority)

Step ("kick-drift-kick"): half kick (velocities: nose rate, speed, angle of attack,
power; pose held) | drift (pose moves along an exact arc; velocities held) | field
evaluation | half kick. The kick is exponential: the fast nose / angle-of-attack
oscillator is solved in closed form, speed and power by RK2.

Storage variants (same physics): body velocity as polar (speed, alpha) or cartesian;
heading as a (cos, sin) pair, an angle, or a PGA motor.
"""
import math, os
from dataclasses import dataclass
import jax, jax.numpy as jnp

THRUST_ALONG = os.environ.get("THRUST_ALONG", "velocity")   # "velocity" (dive/climb) or "nose" (today)
CLIMB_DRAG = bool(int(os.environ.get("CLIMB_DRAG", "0")))     # induced drag from climb lift


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
STICK = jnp.array([0.0, -1 / 3, 1 / 3, -2 / 3, 2 / 3, -1.0, 1.0])


def softmin4(a, b):
    """Smooth min of two non-negative magnitudes (p = 4)."""
    x2 = (a / jnp.maximum(b, 1e-12)) ** 2
    return a / jnp.sqrt(jnp.sqrt(1.0 + x2 * x2))


def lift(sh, alpha):
    x = jnp.clip(alpha / sh.alpha_s, -2.0, 2.0)
    return sh.cl_max * (2.0 * x - x * jnp.abs(x))


def lift_slope(sh, alpha):
    x = alpha / sh.alpha_s
    return jnp.where(jnp.abs(x) < 2.0, sh.cl_max / sh.alpha_s * (2.0 - 2.0 * jnp.abs(x)), 0.0)


def budget(sh, u):
    return softmin4(sh.cl_max * u * u, sh.a_max)


def alpha_cmd(sh, stick, u):
    a = softmin4(jnp.abs(stick) * sh.cl_max * u * u, sh.a_max)
    c = jnp.clip(a / jnp.maximum(u * u, sh.eps) / sh.cl_max, 0.0, 1.0)
    return jnp.sign(stick) * sh.alpha_s * (1.0 - jnp.sqrt(1.0 - c))


def thrust_eff(sh, T, power, u, a_lat):
    """Forward: soft fade at an empty store. Reverse (climb): soft fade at a full store and
    capped by the lift left after turning."""
    band = sh.band * sh.max_power
    L = budget(sh, u)
    left = L * jnp.sqrt(jnp.clip(1.0 - (a_lat / jnp.maximum(L, 1e-12)) ** 2, 0.0, 1.0))
    rev = -softmin4(-jnp.minimum(T, 0.0), left) * jnp.clip((sh.max_power - power) / band, 0.0, 1.0)
    fwd = T * jnp.clip(power / band, 0.0, 1.0)
    return jnp.where(T > 0, fwd, rev)


# ------------------------------------------------------------------ fields
def fields_eval(fx, fy, fr, fwid, flog, x, y):
    dx, dy = x[:, None] - fx[None], y[:, None] - fy[None]
    d = jnp.sqrt(dx * dx + dy * dy)
    zu = 0.5 - (d - fr[None]) / fwid[None]
    z = jnp.clip(zu, 0.0, 1.0)
    prof = z * z * z * (z * (z * 6.0 - 15.0) + 10.0)
    dprof = 30.0 * z * z * (z - 1.0) ** 2
    fac = jnp.where((zu > 0) & (zu < 1) & (d > 1e-6), -dprof / fwid[None] / jnp.maximum(d, 1e-6), 0.0) * flog[None]
    return jnp.exp((prof * flog[None]).sum(1)), (fac * dx).sum(1), (fac * dy).sum(1)


# ------------------------------------------------------------------ the kick (polar core)
def kick_polar(sh, u0, alpha0, ca0, sa0, r, power, n, gbx, gby, T, stick, h):
    """Half kick on (r, speed u, angle of attack alpha, power) with the pose held.
    ca0, sa0 = cos, sin of alpha0 (passed in so storage variants can avoid recomputing).
    Body-frame velocity direction is (cos alpha, -sin alpha). Returns u1, alpha1, r1, power1."""
    along_v = THRUST_ALONG == "velocity"

    def speed_rates(uu, pp, aa, caa):
        cl = lift(sh, aa)
        a_lat = jnp.abs(cl) * uu * uu
        te = thrust_eff(sh, T, pp, uu, a_lat)
        climb = jnp.where(CLIMB_DRAG & (te < 0), te / jnp.maximum(uu * uu, sh.eps), 0.0)
        cd = sh.cd0 + sh.k_ind * (cl * cl + climb * climb)
        along = te if along_v else te * caa
        return (along - cd * uu * uu) / n, sh.regen - along * uu / sh.psc / n

    du0, dp0 = speed_rates(u0, power, alpha0, ca0)
    u = jnp.maximum(u0 + 0.5 * h * du0, sh.eps)              # predicted midpoint speed
    pm = power + 0.5 * h * dp0
    te = thrust_eff(sh, T, pm, u, jnp.abs(lift(sh, alpha0)) * u * u)
    lat_t = 0.0 if along_v else te * sa0                      # nose thrust has a lateral part
    lat_k = 0.0 if along_v else te * ca0
    omega_f = (u / n) * (ca0 * gby + sa0 * gbx)               # v_hat x g with v_hat = (cos a, -sin a)
    theta_dot0 = (lift(sh, alpha0) * u * u + lat_t) / (u * n) + omega_f
    kappa = (lift_slope(sh, alpha0) * u * u + lat_k) / (u * n)
    Om = jnp.maximum(sh.w_c * u / sh.u_corner, sh.w_floor) / n
    d_alpha = alpha_cmd(sh, stick, u) - alpha0
    a11, a21, a22 = -kappa, -Om * Om + 2.0 * sh.zeta * Om * kappa, -2.0 * sh.zeta * Om
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
    alpha1 = alpha0 + e_s + em * (C * y1 + Sq * ((a11 - m) * y1 + y2))
    r1 = theta_dot0 + z_s + em * (C * y2 + Sq * (a21 * y1 + (a22 - m) * y2))
    am = 0.5 * (alpha0 + alpha1)
    cam = jnp.cos(am) if not along_v else ca0
    du, dp = speed_rates(u0, power, am, cam)
    du, dp = speed_rates(jnp.maximum(u0 + 0.5 * h * du, sh.eps), power + 0.5 * h * dp, am, cam)
    return jnp.maximum(u0 + h * du, sh.eps), alpha1, r1, jnp.clip(power + h * dp, 0.0, sh.max_power)


def drift_index(n, gbx, gby, vbx, vby, h):
    return n * jnp.exp(0.5 * h * (gbx * vbx + gby * vby))     # n at the step midpoint (v = w / n)


def arc(r, h):
    """Rotation (cos, sin) of phi = r h and the arc factors (a1, a2): displacement = v (a1 + i a2)."""
    phi = r * h
    cp, sp = jnp.cos(phi), jnp.sin(phi)
    small = jnp.abs(phi) < 1e-4
    a1 = jnp.where(small, 1.0 - phi * phi / 6.0, sp / jnp.where(small, 1.0, phi)) * h
    a2 = jnp.where(small, 0.5 * phi, (1.0 - cp) / jnp.where(small, 1.0, phi)) * h
    return cp, sp, a1, a2


# ------------------------------------------------------------------ storage variants
# 1) x/y, heading (c, s), body velocity polar (u, alpha)
def step_cs_polar(sh, field, st, act, h):
    x, y, c, s, r, u, al, power, n, gx, gy = st
    T, stick = THRUST[act[..., 0]].astype(x.dtype), STICK[act[..., 1]].astype(x.dtype)
    gbx, gby = c * gx + s * gy, -s * gx + c * gy
    u, al, r, power = kick_polar(sh, u, al, jnp.cos(al), jnp.sin(al), r, power, n, gbx, gby, T, stick, 0.5 * h)
    ca, sa = jnp.cos(al), jnp.sin(al)
    nm = drift_index(n, gbx, gby, u * ca / n, -u * sa / n, h)
    vx, vy = u * ca / nm, -u * sa / nm
    cp, sp, a1, a2 = arc(r, h)
    bx, by = vx * a1 - vy * a2, vx * a2 + vy * a1
    x, y = x + c * bx - s * by, y + s * bx + c * by
    c, s = c * cp - s * sp, s * cp + c * sp
    n, gx, gy = fields_eval(*field, x, y)
    gbx, gby = c * gx + s * gy, -s * gx + c * gy
    u, al, r, power = kick_polar(sh, u, al, ca, sa, r, power, n, gbx, gby, T, stick, 0.5 * h)
    return x, y, c, s, r, u, al, power, n, gx, gy


# 2) x/y, heading (c, s), body velocity cartesian (wx, wy): atan2 + sqrt per kick
def step_cs_cart(sh, field, st, act, h):
    x, y, c, s, r, wx, wy, power, n, gx, gy = st
    T, stick = THRUST[act[..., 0]].astype(x.dtype), STICK[act[..., 1]].astype(x.dtype)
    def k(wx, wy, r, power, n, gbx, gby):
        u = jnp.sqrt(wx * wx + wy * wy); us = jnp.maximum(u, sh.eps)
        u, al, r, power = kick_polar(sh, u, jnp.arctan2(-wy, wx), wx / us, -wy / us, r, power, n, gbx, gby, T, stick, 0.5 * h)
        return u * jnp.cos(al), -u * jnp.sin(al), r, power
    gbx, gby = c * gx + s * gy, -s * gx + c * gy
    wx, wy, r, power = k(wx, wy, r, power, n, gbx, gby)
    nm = drift_index(n, gbx, gby, wx / n, wy / n, h)
    vx, vy = wx / nm, wy / nm
    cp, sp, a1, a2 = arc(r, h)
    bx, by = vx * a1 - vy * a2, vx * a2 + vy * a1
    x, y = x + c * bx - s * by, y + s * bx + c * by
    c, s = c * cp - s * sp, s * cp + c * sp
    n, gx, gy = fields_eval(*field, x, y)
    gbx, gby = c * gx + s * gy, -s * gx + c * gy
    wx, wy, r, power = k(wx, wy, r, power, n, gbx, gby)
    return x, y, c, s, r, wx, wy, power, n, gx, gy


# 3) x/y, heading as an angle psi, body velocity polar
def step_angle_polar(sh, field, st, act, h):
    x, y, psi, r, u, al, power, n, gx, gy = st
    T, stick = THRUST[act[..., 0]].astype(x.dtype), STICK[act[..., 1]].astype(x.dtype)
    c, s = jnp.cos(psi), jnp.sin(psi)
    gbx, gby = c * gx + s * gy, -s * gx + c * gy
    u, al, r, power = kick_polar(sh, u, al, jnp.cos(al), jnp.sin(al), r, power, n, gbx, gby, T, stick, 0.5 * h)
    ca, sa = jnp.cos(al), jnp.sin(al)
    nm = drift_index(n, gbx, gby, u * ca / n, -u * sa / n, h)
    vx, vy = u * ca / nm, -u * sa / nm
    _, _, a1, a2 = arc(r, h)
    bx, by = vx * a1 - vy * a2, vx * a2 + vy * a1
    x, y = x + c * bx - s * by, y + s * bx + c * by
    psi = psi + r * h
    c, s = jnp.cos(psi), jnp.sin(psi)
    n, gx, gy = fields_eval(*field, x, y)
    gbx, gby = c * gx + s * gy, -s * gx + c * gy
    u, al, r, power = kick_polar(sh, u, al, ca, sa, r, power, n, gbx, gby, T, stick, 0.5 * h)
    return x, y, psi, r, u, al, power, n, gx, gy


# 4) PGA motor pose, body velocity polar
def motor_mul(m1, m2):
    s, p, a, b = m1
    t, q, c, d = m2
    return (s * t - p * q, p * t + q * s, a * t - b * q + c * s + d * p, a * q + b * t - c * p + d * s)


def motor_exp(r, vx, vy, h):
    """exp(-1/2 (r e12 + vx e01 + vy e02) h) = cos(rh/2) + sinc(rh/2) B (B^2 = -(rh/2)^2 in 2D PGA)."""
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


def step_pga_polar(sh, field, st, act, h):
    m, r, u, al, power, n, gx, gy = st
    T, stick = THRUST[act[..., 0]].astype(power.dtype), STICK[act[..., 1]].astype(power.dtype)
    c, s = motor_heading(m)
    gbx, gby = c * gx + s * gy, -s * gx + c * gy
    u, al, r, power = kick_polar(sh, u, al, jnp.cos(al), jnp.sin(al), r, power, n, gbx, gby, T, stick, 0.5 * h)
    ca, sa = jnp.cos(al), jnp.sin(al)
    nm = drift_index(n, gbx, gby, u * ca / n, -u * sa / n, h)
    m = motor_mul(m, motor_exp(r, u * ca / nm, -u * sa / nm, h))
    x, y = motor_position(m)
    n, gx, gy = fields_eval(*field, x, y)
    c, s = motor_heading(m)
    gbx, gby = c * gx + s * gy, -s * gx + c * gy
    u, al, r, power = kick_polar(sh, u, al, ca, sa, r, power, n, gbx, gby, T, stick, 0.5 * h)
    return m, r, u, al, power, n, gx, gy


def init(variant, x0, y0, head, speed, power, field, dt):
    n, gx, gy = fields_eval(*[f.astype(dt) for f in field], x0.astype(dt), y0.astype(dt))
    z = jnp.zeros_like(x0, dt); u = (speed * n).astype(dt); p = power.astype(dt)
    x0, y0, head = x0.astype(dt), y0.astype(dt), head.astype(dt)
    if variant == "cs_polar":
        return (x0, y0, jnp.cos(head), jnp.sin(head), z, u, z, p, n, gx, gy)
    if variant == "cs_cart":
        return (x0, y0, jnp.cos(head), jnp.sin(head), z, u, z, p, n, gx, gy)
    if variant == "angle_polar":
        return (x0, y0, head, z, u, z, p, n, gx, gy)
    m = motor_mul((1.0 + z, z, -0.5 * x0, -0.5 * y0), (jnp.cos(0.5 * head), -jnp.sin(0.5 * head), z, z))
    return (m, z, u, z, p, n, gx, gy)


def pose(variant, st):
    if variant in ("cs_polar", "cs_cart"):
        return st[0], st[1], jnp.arctan2(st[3], st[2])
    if variant == "angle_polar":
        return st[0], st[1], st[2]
    x, y = motor_position(st[0]); c, s = motor_heading(st[0])
    return x, y, jnp.arctan2(s, c)


STEPS = {"cs_polar": step_cs_polar, "cs_cart": step_cs_cart, "angle_polar": step_angle_polar, "pga_polar": step_pga_polar}
