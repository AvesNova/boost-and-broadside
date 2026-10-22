"""Flight physics v4, shared by the x/y and PGA versions (scalar math only; no geometry).

Controls: stick (7 levels) commands a lateral lift fraction; the power action commands the
vertical part of the same lift vector: coast = hold level, boost = push (dive), reverse =
pull (climb). One lift vector, coupled (p = 2): if the commanded vector exceeds the lift
budget L(u) = softmin(C_Lmax u^2, a_max) it is scaled down along its direction. Holding
level costs g_hold of vertical lift, so below ~sqrt(g_hold / C_Lmax) the ship cannot
hold level and descends (stall nose-drop from gravity). Dive/climb exchange speed with the
power store (altitude) along the flight path: a_x = k (g_hold - L_vert). Engine thrust
T_eng acts along the nose and costs nothing (regen stays as today). Induced drag uses the
whole lift vector. Fields: proper time and proper velocity (see v2).
"""
import math
from dataclasses import dataclass
import jax.numpy as jnp


@dataclass(frozen=True)
class Ship:
    cl_max: float = 0.027
    alpha_s: float = math.radians(15.0)
    cd0: float = 8e-4
    k_ind: float = (5e-3 - 8e-4) / 0.027 ** 2
    a_max: float = 270.0
    g_hold: float = 27.0          # 10% of a_max (assumption, flagged)
    k_x: float = 0.3              # full push at high speed ~ 0.3 (27 + ~250) ~ 80 m/s^2
    t_eng: float = 8.0
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


STICK = jnp.array([0.0, -1 / 3, 1 / 3, -2 / 3, 2 / 3, -1.0, 1.0])
VERT = jnp.array([0.0, -1.0, 1.0])      # coast: level, boost: push, reverse: pull


def softmin4(a, b):
    x2 = (a / jnp.maximum(b, 1e-12)) ** 2
    return a / jnp.sqrt(jnp.sqrt(1.0 + x2 * x2))


def lift(sh, alpha):
    x = jnp.clip(alpha / sh.alpha_s, -2.0, 2.0)
    return sh.cl_max * (2.0 * x - x * jnp.abs(x))


def lift_slope(sh, alpha):
    x = alpha / sh.alpha_s
    return jnp.where(jnp.abs(x) < 2.0, sh.cl_max / sh.alpha_s * (2.0 - 2.0 * jnp.abs(x)), 0.0)


def lift_vector(sh, stick, vmode, u):
    """Commanded (lateral magnitude with sign, vertical) lift, scaled into the budget."""
    L = softmin4(sh.cl_max * u * u, sh.a_max)
    lat = stick * L
    vert = jnp.where(vmode == 0.0, sh.g_hold, vmode * L)
    norm = jnp.sqrt(lat * lat + vert * vert)
    scale = softmin4(norm, L) / jnp.maximum(norm, 1e-12)
    return lat * scale, vert * scale


def alpha_for(sh, lat, u):
    c = jnp.clip(jnp.abs(lat) / jnp.maximum(u * u, sh.eps) / sh.cl_max, 0.0, 1.0)
    return jnp.sign(lat) * sh.alpha_s * (1.0 - jnp.sqrt(1.0 - c))


def exchange(sh, vert, power, u):
    """Dive (+) / climb (-) acceleration along the path, with soft store limits."""
    band = sh.band * sh.max_power
    a_x = sh.k_x * (sh.g_hold - vert)
    fade = jnp.where(a_x > 0, jnp.clip(power / band, 0.0, 1.0), jnp.clip((sh.max_power - power) / band, 0.0, 1.0))
    return a_x * fade


def kick_core(sh, u0, ca0, sa0, alpha0, r, power, n, gbx, gby, stick, vmode, h):
    """Exponential half kick. Body velocity direction is (cos a, -sin a). Returns
    (d_alpha, r1, u1, power1): the velocity is rotated by -d_alpha in the body frame."""
    # the lift vector split is evaluated once per half kick, at the predicted midpoint speed
    lat0, vert0 = lift_vector(sh, stick, vmode, u0)

    def speed_rates(uu, pp, aa, caa, vert):
        cl = lift(sh, aa); cv = vert / jnp.maximum(uu * uu, sh.eps)
        cd = sh.cd0 + sh.k_ind * (cl * cl + cv * cv)
        ax = exchange(sh, vert, pp, uu)
        return (sh.t_eng * caa + ax - cd * uu * uu) / n, sh.regen - ax * uu / sh.psc / n

    du0, dp0 = speed_rates(u0, power, alpha0, ca0, vert0)
    u = jnp.maximum(u0 + 0.5 * h * du0, sh.eps)
    omega_f = (u / n) * (ca0 * gby + sa0 * gbx)
    theta_dot0 = (lift(sh, alpha0) * u * u + sh.t_eng * sa0) / (u * n) + omega_f
    kappa = (lift_slope(sh, alpha0) * u * u + sh.t_eng * ca0) / (u * n)
    Om = jnp.maximum(sh.w_c * u / sh.u_corner, sh.w_floor) / n
    lat, vert = lift_vector(sh, stick, vmode, u)
    d_alpha = alpha_for(sh, lat, u) - alpha0
    a11, a21, a22 = -kappa, -Om * Om + 2.0 * sh.zeta * Om * kappa, -2.0 * sh.zeta * Om
    e_s, z_s = d_alpha, kappa * d_alpha
    y1, y2 = -e_s, (r - theta_dot0) - z_s
    m = 0.5 * (a11 + a22)
    disc = m * m - Om * Om
    q = jnp.sqrt(jnp.abs(disc)); qh = q * h
    osc = disc < 0
    C = jnp.where(osc, jnp.cos(qh), jnp.cosh(jnp.minimum(qh, 30.0)))
    Sq = jnp.where(qh > 1e-6, jnp.where(osc, jnp.sin(qh), jnp.sinh(jnp.minimum(qh, 30.0))) / jnp.maximum(q, 1e-12), h)
    em = jnp.exp(m * h)
    e1 = e_s + em * (C * y1 + Sq * ((a11 - m) * y1 + y2))
    r1 = theta_dot0 + z_s + em * (C * y2 + Sq * (a21 * y1 + (a22 - m) * y2))
    am = alpha0 + 0.5 * e1
    cam = jnp.cos(am)
    du, dp = speed_rates(u0, power, am, cam, vert)
    du, dp = speed_rates(jnp.maximum(u0 + 0.5 * h * du, sh.eps), power + 0.5 * h * dp, am, cam, vert)
    return e1, r1, jnp.maximum(u0 + h * du, sh.eps), jnp.clip(power + h * dp, 0.0, sh.max_power)


def fields_eval(fx, fy, fr, fwid, flog, x, y):
    dx, dy = x[:, None] - fx[None], y[:, None] - fy[None]
    d = jnp.sqrt(dx * dx + dy * dy)
    zu = 0.5 - (d - fr[None]) / fwid[None]
    z = jnp.clip(zu, 0.0, 1.0)
    prof = z * z * z * (z * (z * 6.0 - 15.0) + 10.0)
    dprof = 30.0 * z * z * (z - 1.0) ** 2
    fac = jnp.where((zu > 0) & (zu < 1) & (d > 1e-6), -dprof / fwid[None] / jnp.maximum(d, 1e-6), 0.0) * flog[None]
    return jnp.exp((prof * flog[None]).sum(1)), (fac * dx).sum(1), (fac * dy).sum(1)
