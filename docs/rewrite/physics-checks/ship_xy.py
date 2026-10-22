"""v4 in the x/y (complex-number) paradigm: every planar quantity is a complex number held
as a real pair; rotations are complex multiplications.

State: position p, heading q (unit complex), body velocity w (proper, complex), nose rate
r, power, and the field (n, grad ln n) at p.
"""
import jax.numpy as jnp
import flight4 as F


# complex helpers on (re, im) pairs
def cmul(a, b): return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]
def conj(a): return a[0], -a[1]
def cscale(a, k): return a[0] * k, a[1] * k
def cadd(a, b): return a[0] + b[0], a[1] + b[1]
def cabs(a): return jnp.sqrt(a[0] * a[0] + a[1] * a[1])
def cis(phi): return jnp.cos(phi), jnp.sin(phi)


def arc(phi, h):
    """(e^{i phi} - 1) / (i phi) * h, the displacement factor of a uniform turn."""
    small = jnp.abs(phi) < 1e-4
    ps = jnp.where(small, 1.0, phi)
    re = jnp.where(small, 1.0 - phi * phi / 6.0, jnp.sin(phi) / ps)
    im = jnp.where(small, 0.5 * phi, (1.0 - jnp.cos(phi)) / ps)
    return re * h, im * h


def kick(sh, w, r, power, n, g_body, stick, vmode, h):
    u = cabs(w)
    us = jnp.maximum(u, sh.eps)
    ca, sa = w[0] / us, -w[1] / us                    # w = u e^{-i alpha}
    alpha = jnp.arctan2(sa, ca)
    e1, r1, u1, p1 = F.kick_core(sh, u, ca, sa, alpha, r, power, n, g_body[0], g_body[1], stick, vmode, h)
    w = cscale(cmul(w, cis(-e1)), u1 / us)            # rotate by -d_alpha, rescale
    return w, r1, p1


def step(sh, field, st, act, h):
    p, q, w, r, power, n, g = st
    stick, vmode = F.STICK[act[..., 1]].astype(power.dtype), F.VERT[act[..., 0]].astype(power.dtype)
    w, r, power = kick(sh, w, r, power, n, cmul(conj(q), g), stick, vmode, 0.5 * h)
    g_body = cmul(conj(q), g)
    n_mid = n * jnp.exp(0.5 * h * (g_body[0] * w[0] + g_body[1] * w[1]) / n)
    p = cadd(p, cmul(q, cmul(cscale(w, 1.0 / n_mid), arc(r * h, h))))
    q = cmul(q, cis(r * h))
    n, gx, gy = F.fields_eval(*field, p[0], p[1])
    g = (gx, gy)
    w, r, power = kick(sh, w, r, power, n, cmul(conj(q), g), stick, vmode, 0.5 * h)
    return p, q, w, r, power, n, g


def init(x, y, head, speed, power, field):
    n, gx, gy = F.fields_eval(*field, x, y)
    z = 0.0 * x
    return (x, y), (jnp.cos(head), jnp.sin(head)), (speed * n, z), z, power, n, (gx, gy)


def pose(st):
    return st[0][0], st[0][1], jnp.arctan2(st[1][1], st[1][0])


def energy(sh, st):
    w = st[2]
    return 0.5 * (w[0] ** 2 + w[1] ** 2) + sh.psc * st[4]
