"""2D rigid-body ship prototype (pitch axis in the plane), in x/y and in 2D PGA. JAX.

Units follow today's normalized flight model (mass 1, coefficients per metre):
  lateral accel  = C_L(alpha) u^2,   drag accel = (C_D0 + k C_L^2) u^2,   thrust accel = T
Rotation: the nose psi is a rigid body with aerodynamic stiffness and alpha-dot damping,
  r' = w(u)^2 (alpha_cmd - alpha) - 2 zeta w(u) alpha',   w(u) = w_c u / u_corner,
  alpha = psi - theta (theta = flight-path heading), alpha' = r - theta'.
Stick levels ask for a lift fraction; a soft g limit caps it; alpha_cmd inverts the lift curve.
Power: soft fade over a band at empty (thrust) and full (reverse recovery); regen in the flow.
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
    g_soft_p: float = 4.0
    w_c: float = 12.3          # rad/s pitch frequency at corner speed (reversal ~0.2 s)
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


def lift(sh, alpha):
    x = jnp.clip(alpha / sh.alpha_s, -2.0, 2.0)
    return sh.cl_max * (2.0 * x - x * jnp.abs(x))


def alpha_cmd(sh, stick, u):
    c_req = jnp.abs(stick) * sh.cl_max
    a_w = c_req * u * u
    x2 = (a_w / sh.a_max) ** 2
    a = a_w / jnp.sqrt(jnp.sqrt(1.0 + x2 * x2))   # soft min with p = 4, two sqrts instead of pow
    c = jnp.clip(a / jnp.maximum(u * u, sh.eps) / sh.cl_max, 0.0, 1.0)
    return jnp.sign(stick) * sh.alpha_s * (1.0 - jnp.sqrt(1.0 - c))


def omega(sh, u):
    return jnp.maximum(sh.w_c * u / sh.u_corner, sh.w_floor)


def soft_thrust(sh, T, power):
    band = sh.band * sh.max_power
    fwd = jnp.clip(power / band, 0.0, 1.0)
    rev = jnp.clip((sh.max_power - power) / band, 0.0, 1.0)
    return T * jnp.where(T > 0, fwd, rev)


def path_accel(sh, ux, uy, nx, ny, T, power):
    """World acceleration from thrust (along nose n), lift and drag; and power rate."""
    u = jnp.sqrt(ux * ux + uy * uy)
    dx, dy = ux / jnp.maximum(u, sh.eps), uy / jnp.maximum(u, sh.eps)
    alpha = jnp.arctan2(dx * ny - dy * nx, dx * nx + dy * ny)
    cl = lift(sh, alpha)
    cd = sh.cd0 + sh.k_ind * cl * cl
    Te = soft_thrust(sh, T, power)
    ax = Te * nx - cl * u * u * dy - cd * u * u * dx
    ay = Te * ny + cl * u * u * dx - cd * u * u * dy
    dpower = sh.regen - Te * (ux * nx + uy * ny) / sh.psc   # u cos(alpha) as a dot product
    return ax, ay, dpower, alpha, u


# ------------------------------------------------------------------ x/y version
def _exact_rot(sh, psi, r, theta, theta_dot, a_c, w, h):
    """Exact step of psi'' = w^2 (a_c + theta + theta_dot t... frozen) - 2 zeta w (psi' - theta_dot)."""
    # shift: y = psi - theta - a_c - theta_dot*t ; z = r - theta_dot  ->  y' = z, z' = -w^2 y - 2 zeta w z
    y0, z0 = psi - theta - a_c, r - theta_dot
    zw = sh.zeta * w
    disc = zw * zw - w * w                       # < 0 for zeta < 1
    wd = jnp.sqrt(jnp.maximum(-disc, 0.0))
    e = jnp.exp(-zw * h)
    c, s = jnp.cos(wd * h), jnp.sin(wd * h)
    sinc = jnp.where(wd * h > 1e-6, s / jnp.maximum(wd, 1e-12), h)
    y1 = e * (y0 * c + (z0 + zw * y0) * sinc)
    z1 = e * (z0 * c - (w * w * y0 + zw * z0) * sinc)
    return y1 + theta + a_c + theta_dot * h, z1 + theta_dot


def xy_step(sh, st, act, h):
    px, py, vx, vy, psi, r, power = st
    T, stick = THRUST[act[..., 0]].astype(power.dtype), STICK[act[..., 1]].astype(power.dtype)

    def rot(st, hh):
        px, py, vx, vy, psi, r, power = st
        ax, ay, _, alpha, u = path_accel(sh, vx, vy, jnp.cos(psi), jnp.sin(psi), T, power)
        theta = jnp.arctan2(vy, vx)
        theta_dot = (vx * ay - vy * ax) / jnp.maximum(u * u, sh.eps)
        theta = psi - alpha  # unwrapped path heading consistent with psi (alpha is wrap-safe)
        psi, r = _exact_rot(sh, psi, r, theta, theta_dot, alpha_cmd(sh, stick, u), omega(sh, u), hh)
        return px, py, vx, vy, psi, r, power

    def vel(st, hh):  # RK2 midpoint, nose fixed
        px, py, vx, vy, psi, r, power = st
        nx, ny = jnp.cos(psi), jnp.sin(psi)
        ax, ay, dp, _, _ = path_accel(sh, vx, vy, nx, ny, T, power)
        mx, my, mp = vx + 0.5 * hh * ax, vy + 0.5 * hh * ay, power + 0.5 * hh * dp
        ax, ay, dp, _, _ = path_accel(sh, mx, my, nx, ny, T, mp)
        return px, py, vx + hh * ax, vy + hh * ay, psi, r, jnp.clip(power + hh * dp, 0.0, sh.max_power)

    st = rot(st, 0.5 * h)
    st = vel(st, 0.5 * h)
    px, py, vx, vy, psi, r, power = st
    st = (px + h * vx, py + h * vy, vx, vy, psi, r, power)
    st = vel(st, 0.5 * h)
    return rot(st, 0.5 * h)


# ------------------------------------------------------------------ PGA version
# motor M = s + p e12 + a e01 + b e02 (pga.py conventions); twist in the body frame.
def pga_mul(m1, m2):
    s, p, a, b = m1
    t, q, c, d = m2
    return (s * t - p * q, p * t + q * s, a * t - b * q + c * s + d * p, a * q + b * t - c * p + d * s)


def pga_exp_twist(r, vx, vy, h):
    """exp(B), B = -1/2 (r e12 + vx e01 + vy e02) h. e12 anticommutes with e01, e02, so
    B^2 = -(r h / 2)^2 and exp(B) = cos(rh/2) + sinc(rh/2) B (exact screw motion)."""
    half = 0.5 * r * h
    sinc = jnp.where(jnp.abs(half) > 1e-6, jnp.sin(half) / jnp.where(jnp.abs(half) > 1e-6, half, 1.0), 1.0)
    return jnp.cos(half), -sinc * half, -0.5 * sinc * vx * h, -0.5 * sinc * vy * h


def pga_apply_origin(m):
    s, p, a, b = m
    return -2.0 * (a * s + b * p), 2.0 * (a * p - b * s)


def pga_heading(m):
    s, p, _, _ = m
    return s * s - p * p, -2.0 * p * s       # (cos psi, sin psi)


def pga_from_pose(x, y, c, sn):
    psi = jnp.arctan2(sn, c)
    rot = (jnp.cos(0.5 * psi), -jnp.sin(0.5 * psi), 0.0 * x, 0.0 * x)
    tr = (1.0 + 0.0 * x, 0.0 * x, -0.5 * x, -0.5 * y)
    return pga_mul(tr, rot)


def body_rates(sh, M, tw, T, stick, power):
    """Twist derivative in the body frame: v_b' = a_b - r x v_b ; r' = aero moment."""
    r, vbx, vby = tw
    c, sn = pga_heading(M)
    wx, wy = c * vbx - sn * vby, sn * vbx + c * vby          # world velocity
    ax, ay, dp, alpha, u = path_accel(sh, wx, wy, c, sn, T, power)
    abx, aby = c * ax + sn * ay, -sn * ax + c * ay            # body acceleration
    theta_dot = (wx * ay - wy * ax) / jnp.maximum(u * u, sh.eps)
    w = omega(sh, u)
    rdot = w * w * (alpha_cmd(sh, stick, u) - alpha) - 2.0 * sh.zeta * w * (r - theta_dot)
    return rdot, abx + r * vby, aby - r * vbx, dp


def pga_step(sh, st, act, h):
    M, tw, power = st
    T, stick = THRUST[act[..., 0]].astype(power.dtype), STICK[act[..., 1]].astype(power.dtype)

    def kick(M, tw, power, hh):  # RK2 midpoint on the twist (pose held)
        k = body_rates(sh, M, tw, T, stick, power)
        mid = tuple(x + 0.5 * hh * dx for x, dx in zip(tw, k[:3]))
        k = body_rates(sh, M, mid, T, stick, power + 0.5 * hh * k[3])
        return tuple(x + hh * dx for x, dx in zip(tw, k[:3])), jnp.clip(power + hh * k[3], 0.0, sh.max_power)

    tw, power = kick(M, tw, power, 0.5 * h)
    M = pga_mul(M, pga_exp_twist(*tw, h))
    tw, power = kick(M, tw, power, 0.5 * h)
    return M, tw, power


# ------------------------------------------------------------------ x/y, same algorithm as PGA
def xy2_rates(sh, c, sn, r, vbx, vby, T, stick, power):
    wx, wy = c * vbx - sn * vby, sn * vbx + c * vby
    ax, ay, dp, alpha, u = path_accel(sh, wx, wy, c, sn, T, power)
    abx, aby = c * ax + sn * ay, -sn * ax + c * ay
    theta_dot = (wx * ay - wy * ax) / jnp.maximum(u * u, sh.eps)
    w = omega(sh, u)
    rdot = w * w * (alpha_cmd(sh, stick, u) - alpha) - 2.0 * sh.zeta * w * (r - theta_dot)
    return rdot, abx + r * vby, aby - r * vbx, dp


def xy2_step(sh, st, act, h):
    """State: position (x, y), heading as a unit pair (c, s), body twist (r, vbx, vby), power."""
    x, y, c, sn, r, vbx, vby, power = st
    T, stick = THRUST[act[..., 0]].astype(power.dtype), STICK[act[..., 1]].astype(power.dtype)

    def kick(tw, power, hh):
        k = xy2_rates(sh, c, sn, *tw, T, stick, power)
        mid = tuple(v + 0.5 * hh * d for v, d in zip(tw, k[:3]))
        k = xy2_rates(sh, c, sn, *mid, T, stick, power + 0.5 * hh * k[3])
        return tuple(v + hh * d for v, d in zip(tw, k[:3])), jnp.clip(power + hh * k[3], 0.0, sh.max_power)

    (r, vbx, vby), power = kick((r, vbx, vby), power, 0.5 * h)
    # exact screw: rotate by phi = r h; body displacement = v_b (e^{i phi} - 1) / (i phi)
    phi = r * h
    small = jnp.abs(phi) < 1e-6
    ps = jnp.where(small, 1.0, phi)
    fx = jnp.where(small, h, jnp.sin(phi) / ps * h)              # Re[(e^{i phi}-1)/(i phi)] h
    fy = jnp.where(small, 0.5 * phi * h, (1 - jnp.cos(phi)) / ps * h)
    dbx, dby = vbx * fx - vby * fy, vbx * fy + vby * fx
    x, y = x + c * dbx - sn * dby, y + sn * dbx + c * dby
    cp, sp = jnp.cos(phi), jnp.sin(phi)
    c, sn = c * cp - sn * sp, c * sp + sn * cp
    (r, vbx, vby), power = kick((r, vbx, vby), power, 0.5 * h)
    return x, y, c, sn, r, vbx, vby, power
