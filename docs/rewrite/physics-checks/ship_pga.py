"""v4 in the 2D PGA paradigm (R(2,0,1)): the pose is a motor M, the velocity is a twist
bivector in the body frame, Omega = r e12 + v01 e01 + v02 e02 (rotation rate and the
translation generator), and the field gradient is an ideal point (a direction).

The kick rotates the twist's translation part with a rotor sandwich; the drift composes
the pose with the motor exponential of the twist; frame changes are motor sandwiches.
All products are hand-expanded from pga_derive.py / pga_derive2.py.
"""
import jax.numpy as jnp
import flight4 as F


# ---- even subalgebra: motor (s, p, a, b) = s + p e12 + a e01 + b e02
def gp(m1, m2):
    s, p, a, b = m1
    t, q, c, d = m2
    return (s * t - p * q, p * t + q * s, a * t - b * q + c * s + d * p, a * q + b * t - c * p + d * s)


def rotor(theta):
    return jnp.cos(0.5 * theta), -jnp.sin(0.5 * theta), 0.0 * theta, 0.0 * theta


def exp_twist(r, v01, v02, h):
    """exp(-h/2 Omega), Omega = r e12 + v01 e01 + v02 e02. In 2D PGA e12 anticommutes with
    e01 and e02, so B^2 = -(r h / 2)^2 and exp(B) = cos(rh/2) + sinc(rh/2) B."""
    half = 0.5 * r * h
    small = jnp.abs(half) < 1e-4
    sinc = jnp.where(small, 1.0 - half * half / 6.0, jnp.sin(half) / jnp.where(small, 1.0, half))
    return jnp.cos(half), -sinc * half, -0.5 * sinc * v01 * h, -0.5 * sinc * v02 * h


# ---- sandwiches on bivectors X = x01 e01 + x02 e02 + x12 e12
def sandwich(m, x01, x02, x12):
    """M X ~M."""
    s, p, a, b = m
    cs, sn = s * s - p * p, 2.0 * p * s
    return (cs * x01 + sn * x02 + 2.0 * (a * p - b * s) * x12,
            cs * x02 - sn * x01 + 2.0 * (a * s + b * p) * x12,
            (s * s + p * p) * x12)


def sandwich_rev(m, x01, x02, x12):
    """~M X M (world to body)."""
    s, p, a, b = m
    cs, sn = s * s - p * p, 2.0 * p * s
    return (cs * x01 - sn * x02 + 2.0 * (a * p + b * s) * x12,
            cs * x02 + sn * x01 + 2.0 * (b * p - a * s) * x12,
            (s * s + p * p) * x12)


# points / directions: x e20 + y e01 + w e12  ->  (e01, e02, e12) = (y, -x, w)
def to_point(x, y, w): return y, -x, w
def from_point(x01, x02, x12): return -x02, x01


def position(m):
    return from_point(*sandwich(m, 0.0, 0.0, 1.0))


def kick(sh, M, tw, power, n, g_body, stick, vmode, h):
    r, v01, v02 = tw                                   # translation generator ~ (vx, vy) in the body
    u = jnp.sqrt(v01 * v01 + v02 * v02)
    us = jnp.maximum(u, sh.eps)
    ca, sa = v01 / us, -v02 / us
    alpha = jnp.arctan2(sa, ca)
    e1, r1, u1, p1 = F.kick_core(sh, u, ca, sa, alpha, r, power, n, g_body[0], g_body[1], stick, vmode, h)
    # rotate the twist's translation part by -d_alpha (rotor sandwich), rescale; set the rate
    v01, v02, _ = sandwich(rotor(-e1), v01, v02, 0.0)
    k = u1 / us
    return (r1, v01 * k, v02 * k), p1


def step(sh, field, st, act, h):
    M, tw, power, n, g = st
    stick, vmode = F.STICK[act[..., 1]].astype(power.dtype), F.VERT[act[..., 0]].astype(power.dtype)
    gb = from_point(*sandwich_rev(M, *to_point(g[0], g[1], 0.0)))
    tw, power = kick(sh, M, tw, power, n, gb, stick, vmode, 0.5 * h)
    r, v01, v02 = tw
    n_mid = n * jnp.exp(0.5 * h * (gb[0] * v01 + gb[1] * v02) / n)
    M = gp(M, exp_twist(r, v01 / n_mid, v02 / n_mid, h))
    x, y = position(M)
    n, gx, gy = F.fields_eval(*field, x, y)
    g = (gx, gy)
    gb = from_point(*sandwich_rev(M, *to_point(gx, gy, 0.0)))
    tw, power = kick(sh, M, tw, power, n, gb, stick, vmode, 0.5 * h)
    return M, tw, power, n, g


def init(x, y, head, speed, power, field):
    n, gx, gy = F.fields_eval(*field, x, y)
    z = 0.0 * x
    T = (1.0 + z, z, -0.5 * x, -0.5 * y)               # translator to (x, y)
    M = gp(T, rotor(head))
    return M, (z, speed * n, z), power, n, (gx, gy)


def pose(st):
    M = st[0]
    x, y = position(M)
    s, p = M[0], M[1]
    return x, y, jnp.arctan2(-2.0 * p * s, s * s - p * p)


def energy(sh, st):
    _, v01, v02 = st[1]
    return 0.5 * (v01 ** 2 + v02 ** 2) + sh.psc * st[2]
