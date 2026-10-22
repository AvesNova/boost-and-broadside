"""Flight v5 (x/y only), retuned toward P-51D proportions at game scale, SI units, 1 kg ship.

Game scale vs a real P-51D: time x4.1, length x6.1 (game faster and smaller), speed x1.48.
State per ship: position (x, y), heading (c, s), speed u and angle of attack alpha (body
velocity in polar form: fastest), nose rate r, energy store E (J), field n and grad ln n.

Engine: constant power, split: P_thrust as thrust along the nose (force P_thrust / u,
softly capped by F_static at low speed) and P_store charging the energy store (wasted when
the store is full). Lift vector (coupled, p = 2): stick = lateral share, coast = hold level
(costs g), boost = push, reverse = pull; dive/climb exchange a_x = k_x (g - L_vert) along
the path, paid from / into E. Induced drag on the whole lift vector. Fields: proper time
and velocity; bending; n at the step midpoint. Step: half kick | arc drift | fields | half kick.
"""
import math
from dataclasses import dataclass
import jax.numpy as jnp


@dataclass(frozen=True)
class Ship:
    mass: float = 1.0               # kg
    g: float = 27.0                 # m/s^2, game gravity ("1 g"): lift needed to hold level
    cl_max: float = 0.027           # 1/m: max lift accel = cl_max u^2  (stall 31.6 m/s)
    alpha_s: float = math.radians(15.0)
    a_max: float = 216.0            # m/s^2 = 8 g
    cd0: float = 2.9e-4             # 1/m: parasitic drag accel = cd0 u^2
    k_ind: float = 3.8              # induced drag accel = k_ind (lift/u^2)^2 u^2
    p_thrust: float = 320.0         # W
    p_store: float = 80.0           # W
    f_static: float = 10.0          # N, thrust cap at low speed
    e_max: float = 10_000.0         # J  (= 370 m of game altitude, ~2.2 km real)
    band: float = 0.02              # soft limits over 2% of the store at each end
    k_x: float = 0.1                # dive/climb exchange per unit of vertical lift change
    w_c: float = 12.3 * math.sqrt(216.0 / 270.0)   # nose frequency at corner speed, rescaled
    zeta: float = 0.7
    w_floor: float = 1.0
    eps: float = 1e-6

    @property
    def u_corner(self):
        return math.sqrt(self.a_max / self.cl_max)


STICK = jnp.array([0.0, -1 / 3, 1 / 3, -2 / 3, 2 / 3, -1.0, 1.0])
VERT = jnp.array([0.0, -1.0, 1.0])      # coast: level, boost: push (dive), reverse: pull (climb)


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
    L = softmin4(sh.cl_max * u * u, sh.a_max)
    lat = stick * L
    vert = jnp.where(vmode == 0.0, sh.g, vmode * L)
    norm = jnp.sqrt(lat * lat + vert * vert)
    scale = softmin4(norm, L) / jnp.maximum(norm, 1e-12)
    return lat * scale, vert * scale


def alpha_for(sh, lat, u):
    c = jnp.clip(jnp.abs(lat) / jnp.maximum(u * u, sh.eps) / sh.cl_max, 0.0, 1.0)
    return jnp.sign(lat) * sh.alpha_s * (1.0 - jnp.sqrt(1.0 - c))


def thrust(sh, u):
    return softmin4(sh.p_thrust / jnp.maximum(u, sh.eps), sh.f_static) / sh.mass


def kick(sh, u0, alpha0, r, E, n, gbx, gby, stick, vmode, h):
    """Exponential half kick on (u, alpha, r, E); pose held. Body velocity = u (cos a, -sin a)."""
    band = sh.band * sh.e_max
    ca0, sa0 = jnp.cos(alpha0), jnp.sin(alpha0)
    lat0, vert0 = lift_vector(sh, stick, vmode, u0)

    def rates(uu, ee, aa, caa, vert):
        cl = lift(sh, aa); cv = vert / jnp.maximum(uu * uu, sh.eps)
        drag = (sh.cd0 + sh.k_ind * (cl * cl + cv * cv)) * uu * uu
        ax = sh.k_x * (sh.g - vert)
        ax = ax * jnp.where(ax > 0, jnp.clip(ee / band, 0.0, 1.0), jnp.clip((sh.e_max - ee) / band, 0.0, 1.0))
        charge = sh.p_store * jnp.clip((sh.e_max - ee) / band, 0.0, 1.0)
        return (thrust(sh, uu) * caa + ax - drag) / n, charge - sh.mass * ax * uu / n

    du0, dE0 = rates(u0, E, alpha0, ca0, vert0)
    u = jnp.maximum(u0 + 0.5 * h * du0, sh.eps)
    lat, vert = lift_vector(sh, stick, vmode, u)
    t = thrust(sh, u)
    omega_f = (u / n) * (ca0 * gby + sa0 * gbx)
    theta_dot0 = (lift(sh, alpha0) * u * u + t * sa0) / (u * n) + omega_f
    kappa = (lift_slope(sh, alpha0) * u * u + t * ca0) / (u * n)
    Om = jnp.maximum(sh.w_c * u / sh.u_corner, sh.w_floor) / n
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
    du, dE = rates(u0, E, am, cam, vert)
    du, dE = rates(jnp.maximum(u0 + 0.5 * h * du, sh.eps), E + 0.5 * h * dE, am, cam, vert)
    return jnp.maximum(u0 + h * du, sh.eps), alpha0 + e1, r1, jnp.clip(E + h * dE, 0.0, sh.e_max)


def fields_eval(fx, fy, fr, fwid, flog, x, y):
    dx, dy = x[:, None] - fx[None], y[:, None] - fy[None]
    d = jnp.sqrt(dx * dx + dy * dy)
    zu = 0.5 - (d - fr[None]) / fwid[None]
    z = jnp.clip(zu, 0.0, 1.0)
    prof = z * z * z * (z * (z * 6.0 - 15.0) + 10.0)
    dprof = 30.0 * z * z * (z - 1.0) ** 2
    fac = jnp.where((zu > 0) & (zu < 1) & (d > 1e-6), -dprof / fwid[None] / jnp.maximum(d, 1e-6), 0.0) * flog[None]
    return jnp.exp((prof * flog[None]).sum(1)), (fac * dx).sum(1), (fac * dy).sum(1)


def step(sh, field, st, act, h):
    x, y, c, s, u, al, r, E, n, gx, gy = st
    stick, vmode = STICK[act[..., 1]].astype(x.dtype), VERT[act[..., 0]].astype(x.dtype)
    gbx, gby = c * gx + s * gy, -s * gx + c * gy
    u, al, r, E = kick(sh, u, al, r, E, n, gbx, gby, stick, vmode, 0.5 * h)
    ca, sa = jnp.cos(al), jnp.sin(al)
    vbx, vby = u * ca / n, -u * sa / n
    n_mid = n * jnp.exp(0.5 * h * (gbx * vbx + gby * vby))
    vbx, vby = vbx * n / n_mid, vby * n / n_mid
    phi = r * h
    cp, sp = jnp.cos(phi), jnp.sin(phi)
    small = jnp.abs(phi) < 1e-4
    ps = jnp.where(small, 1.0, phi)
    a1 = jnp.where(small, 1.0 - phi * phi / 6.0, sp / ps) * h
    a2 = jnp.where(small, 0.5 * phi, (1.0 - cp) / ps) * h
    bx, by = vbx * a1 - vby * a2, vbx * a2 + vby * a1
    x, y = x + c * bx - s * by, y + s * bx + c * by
    c, s = c * cp - s * sp, s * cp + c * sp
    n, gx, gy = fields_eval(*field, x, y)
    gbx, gby = c * gx + s * gy, -s * gx + c * gy
    u, al, r, E = kick(sh, u, al, r, E, n, gbx, gby, stick, vmode, 0.5 * h)
    return x, y, c, s, u, al, r, E, n, gx, gy


def init(x, y, head, speed, E, field):
    n, gx, gy = fields_eval(*field, x, y)
    z = 0.0 * x
    return x, y, jnp.cos(head), jnp.sin(head), speed * n, z, z, E, n, gx, gy


def energy(sh, st):
    return 0.5 * sh.mass * st[4] ** 2 + st[7]
