"""Flight v7 (x/y): commanded energy exchange, no implied angle. (Derived from v6.)

Each power action commands an exchange acceleration a_x along the flight path (coast
+a0 small climb, boost -a_dive, reverse +a_climb); dE/dt = m u a_x and the speed loses a_x.
Reverse must slow the ship at every speed: the engine's static thrust per kg stays below
a_climb. Lift coupling, forced sink and store-end fades as in v6, with a_x in place of
g sin(gamma).

v6 description (for reference): memoryless pitch. All engine power is thrust along the nose; each power
action picks a flight-path pitch (coast +gamma0, boost -gamma_max, reverse +gamma_max), so
gravity along the path trades speed with the energy store (E = m g h, dE/dt = m g u sin g).

Pitch is limited by lift, coupled with turning (p = 2): the commanded lift vector
(lateral = stick x L, vertical = g cos gamma_c to hold the pitch) is scaled into the
budget L(u) = softmin(C_Lmax u^2, a_max). The pitch actually flown is the command scaled by
the lift left after turning, minus a forced nose-down when the vertical lift cannot hold
the pitch (stall, or max turns at low speed). Near a full store, positive pitch fades
(ceiling); near an empty store, negative pitch fades (floor). Shooting costs E per bullet.
Same integrator as v5 fast (exponential nose kick, exact arc, fields as curved space).
"""
import math
from dataclasses import dataclass, replace
import jax, jax.numpy as jnp
from flight5 import fields_eval
from flight5_fast import softmin4, small_cos_sin, expm2


@dataclass(frozen=True)
class Ship:
    mass: float = 1.0
    g: float = 27.0
    cl_max: float = 0.027
    alpha_s: float = math.radians(15.0)
    a_max: float = 216.0
    cd0: float = 2.9e-4
    k_ind: float = 3.8
    power: float = 400.0          # W, all to thrust
    f_static: float = 10.0        # N
    e_max: float = 10_000.0       # J
    band: float = 0.02
    a0: float = 2.0               # m/s^2, coast: small climb (charges the store)
    a_climb: float = 27.0         # m/s^2, reverse
    a_dive: float = 27.0          # m/s^2, boost
    bullet_cost: float = 0.0      # J per bullet
    w_c: float = 12.3 * math.sqrt(216.0 / 270.0)
    zeta: float = 0.7
    w_floor: float = 1.0
    eps: float = 1e-6

    @property
    def u_corner(self):
        return math.sqrt(self.a_max / self.cl_max)


STICK = jnp.array([0.0, -1 / 3, 1 / 3, -2 / 3, 2 / 3, -1.0, 1.0])
PITCH = jnp.array([0.0, -1.0, 1.0])      # coast, boost (dive), reverse (climb)


def lift(sh, alpha):
    x = jnp.clip(alpha / sh.alpha_s, -2.0, 2.0)
    return sh.cl_max * (2.0 * x - x * jnp.abs(x))


def lift_slope(sh, alpha):
    x = alpha / sh.alpha_s
    return jnp.where(jnp.abs(x) < 2.0, sh.cl_max / sh.alpha_s * (2.0 - 2.0 * jnp.abs(x)), 0.0)


def flight_setting(sh, stick, pmode, u, u_inv, E):
    """Lateral lift, exchange acceleration a_x (+ climb), vertical lift, thrust: frozen per half kick."""
    a_c = jnp.where(pmode == 0.0, sh.a0, jnp.where(pmode > 0, sh.a_climb, -sh.a_dive))
    L = softmin4(sh.cl_max * u * u, 1.0 / sh.a_max)
    lat_c, vert_c = stick * L, sh.g
    norm = jnp.sqrt(lat_c * lat_c + vert_c * vert_c)
    scale = softmin4(norm, 1.0 / jnp.maximum(L, 1e-12)) / jnp.maximum(norm, 1e-12)
    lat, vert = lat_c * scale, vert_c * scale
    held = vert / vert_c
    turn_share = jnp.clip(jnp.abs(lat) / jnp.maximum(L, 1e-12), 0.0, 1.0)
    authority = jnp.sqrt(1.0 - turn_share * turn_share)
    a = jnp.where(held > 0.999, a_c * authority, -(1.0 - held) * sh.a_dive - (1.0 - held) * 0.0)  # no blend: full command while lift holds, else sink only
    inv_band = 1.0 / (sh.band * sh.e_max)
    a = a * jnp.where(a > 0, jnp.clip((sh.e_max - E) * inv_band, 0.0, 1.0), jnp.clip(E * inv_band, 0.0, 1.0))
    thrust = softmin4(sh.power * u_inv, 1.0 / sh.f_static) / sh.mass
    return lat, a, vert, thrust


def kick(sh, u0, alpha0, ca0, sa0, r, E, n_inv, gbx, gby, stick, pmode, shoot, h):
    u0s = jnp.maximum(u0, sh.eps)

    def rates(uu, cl, caa, vert, thrust, s, u2_inv):
        cv = vert * u2_inv
        drag = (sh.cd0 + sh.k_ind * (cl * cl + cv * cv)) * uu * uu
        return (thrust * caa - drag - s) * n_inv, sh.mass * uu * s * n_inv

    u0_inv = 1.0 / u0s
    _, s0, vert0, thrust0 = flight_setting(sh, stick, pmode, u0s, u0_inv, E)
    cl0 = lift(sh, alpha0)
    du0, _ = rates(u0, cl0, ca0, vert0, thrust0, s0, u0_inv * u0_inv)
    u = jnp.maximum(u0 + 0.5 * h * du0, sh.eps)
    u_inv = 1.0 / u
    lat, s, vert, t = flight_setting(sh, stick, pmode, u, u_inv, E)
    omega_f = u * n_inv * (ca0 * gby + sa0 * gbx)
    theta_dot0 = (cl0 * u * u + t * sa0) * u_inv * n_inv + omega_f
    kappa = (lift_slope(sh, alpha0) * u * u + t * ca0) * u_inv * n_inv
    Om = jnp.maximum(sh.w_c * u * (1.0 / sh.u_corner), sh.w_floor) * n_inv
    c = jnp.clip(jnp.abs(lat) * u_inv * u_inv * (1.0 / sh.cl_max), 0.0, 1.0)
    d_alpha = jnp.sign(lat) * sh.alpha_s * (1.0 - jnp.sqrt(1.0 - c)) - alpha0
    a11, a21, a22 = -kappa, -Om * Om + 2.0 * sh.zeta * Om * kappa, -2.0 * sh.zeta * Om
    e_s, z_s = d_alpha, kappa * d_alpha
    y1, y2 = -e_s, (r - theta_dot0) - z_s
    m11, m12, m21, m22 = expm2(a11, 1.0, a21, a22, h)
    e1 = e_s + m11 * y1 + m12 * y2
    r1 = theta_dot0 + z_s + m21 * y1 + m22 * y2
    ch, sh_ = small_cos_sin(0.5 * e1)
    cam = ca0 * ch - sa0 * sh_
    clm = lift(sh, alpha0 + 0.5 * e1)
    u2i = u_inv * u_inv
    du, dE = rates(u0, clm, cam, vert, t, s, u2i)
    du, dE = rates(jnp.maximum(u0 + 0.5 * h * du, sh.eps), clm, cam, vert, t, s, u2i)
    c1, s1 = small_cos_sin(e1)
    E1 = E + h * dE
    return (jnp.maximum(u0 + h * du, sh.eps), alpha0 + e1, ca0 * c1 - sa0 * s1, sa0 * c1 + ca0 * s1,
            r1, jnp.clip(E1, 0.0, sh.e_max))


def step(sh, field, st, act, h):
    """act[..., 0] power (0 coast, 1 boost, 2 reverse), [..., 1] stick (7 levels), [..., 2] shoot."""
    x, y, c, s, u, al, ca, sa, r, E, n, gx, gy = st
    stick, pmode = STICK[act[..., 1]].astype(x.dtype), PITCH[act[..., 0]].astype(x.dtype)
    shoot = act[..., 2].astype(x.dtype) * (E >= sh.bullet_cost)  # a bullet needs its energy
    E = E - shoot * sh.bullet_cost                               # one bullet per decision step
    n_inv = 1.0 / n
    gbx, gby = c * gx + s * gy, -s * gx + c * gy
    u, al, ca, sa, r, E = kick(sh, u, al, ca, sa, r, E, n_inv, gbx, gby, stick, pmode, shoot, 0.5 * h)
    vbx, vby = u * ca * n_inv, -u * sa * n_inv
    xg = 0.5 * h * (gbx * vbx + gby * vby)
    k = 1.0 / (1.0 + xg * (1.0 + xg * (0.5 + xg / 6.0)))
    vbx, vby = vbx * k, vby * k
    phi = r * h
    cp, sp = small_cos_sin(phi)
    p2 = phi * phi
    a1 = (1.0 - p2 * (1.0 / 6.0 - p2 / 120.0)) * h
    a2 = phi * (0.5 - p2 * (1.0 / 24.0 - p2 / 720.0)) * h
    bx, by = vbx * a1 - vby * a2, vbx * a2 + vby * a1
    x, y = x + c * bx - s * by, y + s * bx + c * by
    c, s = c * cp - s * sp, s * cp + c * sp
    n, gx, gy = fields_eval(*field, x, y)
    n_inv = 1.0 / n
    gbx, gby = c * gx + s * gy, -s * gx + c * gy
    u, al, ca, sa, r, E = kick(sh, u, al, ca, sa, r, E, n_inv, gbx, gby, stick, pmode, shoot, 0.5 * h)
    return x, y, c, s, u, al, ca, sa, r, E, n, gx, gy


def init(x, y, head, speed, E, field):
    n, gx, gy = fields_eval(*field, x, y)
    z = 0.0 * x
    return x, y, jnp.cos(head), jnp.sin(head), speed * n, z, 1.0 + z, z, z, E, n, gx, gy
