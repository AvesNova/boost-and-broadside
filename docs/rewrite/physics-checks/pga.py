"""Minimal 2D PGA (R(2,0,1)) helpers in JAX, struct-of-arrays, hand-expanded.

Motor M = s + p e12 + a e01 + b e02 (formulas from pga_derive.py).
Points are x e20 + y e01 + w e12 (w = 1); directions (ideal points) have w = 0.
"""
import jax.numpy as jnp


def rotor(theta):
    """Rotation by +theta about the origin."""
    return jnp.cos(0.5 * theta), -jnp.sin(0.5 * theta), 0.0, 0.0


def translator(tx, ty):
    return 1.0, 0.0, -0.5 * tx, -0.5 * ty


def mul(m1, m2):
    """Geometric product m1 m2 (apply m2 first, then m1)."""
    s, p, a, b = m1
    t, q, c, d = m2
    return (s * t - p * q, p * t + q * s, a * t - b * q + c * s + d * p, a * q + b * t - c * p + d * s)


def apply_point(m, x, y):
    s, p, a, b = m
    cs, sn = s * s - p * p, 2.0 * p * s
    return cs * x + sn * y - 2.0 * (a * s + b * p), cs * y - sn * x + 2.0 * (a * p - b * s)


def apply_dir(m, x, y):
    s, p, _, _ = m
    cs, sn = s * s - p * p, 2.0 * p * s
    return cs * x + sn * y, cs * y - sn * x


def normalize(m):
    s, p, a, b = m
    n = jnp.sqrt(s * s + p * p)
    return s / n, p / n, a / n, b / n
