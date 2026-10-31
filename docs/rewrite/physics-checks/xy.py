"""Complex-style helpers over separate real x/y arrays (JAX), the counterpart of pga.py."""
import jax.numpy as jnp


def polar(angle):
    return jnp.cos(angle), jnp.sin(angle)


def mul(ax, ay, bx, by):
    """(a)(b) as complex numbers: rotate a by the unit b, or compose two rotations."""
    return ax * bx - ay * by, ax * by + ay * bx


def conj(x, y):
    return x, -y


def norm(x, y):
    return jnp.sqrt(x * x + y * y)


def normalize(x, y, eps=1e-6):
    n = jnp.maximum(norm(x, y), eps)
    return x / n, y / n


def angle(x, y):
    return jnp.arctan2(y, x)
