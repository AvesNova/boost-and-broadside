"""Comparison and state-building helpers for the acceptance tests."""

import math

import numpy as np

from race_presets import FLIGHT, PRESETS, reset_uniform_count

H = 1.0 / 30.0

# Absolute tolerances per canonical key for one-tick comparisons (float32 vs float64 truth).
ATOL = dict(x=2e-3, y=2e-3, c=2e-5, s=2e-5, w=2e-3, alpha=2e-5, ca=2e-5, sa=2e-5, r=2e-4, E=5e-2,
            logn=2e-6, gx=2e-6, gy=2e-6, shield=1e-3, shield_wait=1e-5,
            bx=2e-3, by=2e-3, bdx=2e-5, bdy=2e-5, bw=2e-3, blogn=2e-6, bgx=2e-6, bgy=2e-6,
            fx=1e-2, fy=1e-2, fr=1e-2, fw=1e-5, axis_x=1e-6, axis_y=1e-6,
            damage=1e-3, boundary_damage=1e-4)


EPS32 = float(np.finfo(np.float32).eps)
HESSIAN_MAX = 3e-3  # bound on |second derivative of ln n| (1/m^2): level 2 ln sqrt2 * 5.8 / (40 m)^2


def medium_atol(x, y, gx=None, gy=None):
    """Tolerance on ln n (with its gradient) or on its gradient (without): a float32 arm stores
    positions rounded to ~eps32 |p|, and the medium at the rounded point differs by the
    gradient (or the second derivative) times that rounding, which grows across a large map."""
    resolution = 4.0 * EPS32 * np.maximum(np.maximum(np.abs(x), np.abs(y)), 1.0)
    slope = np.hypot(gx, gy) if gx is not None else HESSIAN_MAX
    return 2e-6 + resolution * slope


MEDIUM = {"logn": ("x", "y", "gx", "gy"), "blogn": ("bx", "by", "bgx", "bgy")}
GRADIENT = {"gx": ("x", "y"), "gy": ("x", "y"), "bgx": ("bx", "by"), "bgy": ("bx", "by")}


def compare(actual, expected, key, *, atol=None, rtol=1e-5, max_bad=0.0, context=""):
    """Assert |a - e| <= atol + rtol |e| elementwise, allowing a fraction max_bad of misses.

    Discrete arrays (bool/int) compare exactly under the same miss budget.
    """
    a = np.asarray(actual)
    e = np.asarray(expected)
    assert a.shape == e.shape, f"{context}{key}: shape {a.shape} != expected {e.shape}"
    if e.dtype.kind in "biu":
        bad = a.astype(np.int64) != e.astype(np.int64)
        worst = ""
    else:
        tol = (ATOL.get(key, 1e-5) if atol is None else atol) + rtol * np.abs(e)
        err = np.abs(a.astype(np.float64) - e.astype(np.float64))
        bad = ~(err <= tol)
        worst = f", worst error {np.nanmax(err) if err.size else 0:.3g}"
    n_bad = int(bad.sum())
    allowed = int(math.floor(max_bad * bad.size))
    if n_bad > allowed:
        idx = np.argwhere(bad)[:5]
        samples = "; ".join(f"{tuple(i)}: got {a[tuple(i)]}, want {e[tuple(i)]}" for i in idx)
        raise AssertionError(f"{context}{key}: {n_bad} of {bad.size} entries out of tolerance "
                             f"(allowed {allowed}){worst}. e.g. {samples}")


def compare_dict(actual: dict, expected: dict, *, max_bad=0.0, rtol=1e-5, context="", skip=()):
    missing = [k for k in expected if k not in actual]
    assert not missing, f"{context}missing keys {missing}"
    for k, e in expected.items():
        if k in skip:
            continue
        atol = None
        if k in MEDIUM and all(n in expected for n in MEDIUM[k]):
            atol = medium_atol(*(np.asarray(expected[n], np.float64) for n in MEDIUM[k]))
        elif k in GRADIENT and all(n in expected for n in GRADIENT[k]):
            atol = medium_atol(*(np.asarray(expected[n], np.float64) for n in GRADIENT[k]))
        compare(actual[k], e, k, atol=atol, rtol=rtol, max_bad=max_bad, context=context)


def far_fields(state):
    """Move every field of a canonical state out of play (no medium near the origin)."""
    st = {k: v.copy() for k, v in state.items()}
    B, M = st["fx"].shape
    st["fx"] = np.broadcast_to(1e5 + 3000.0 * np.arange(M, dtype=np.float32), (B, M)).copy()
    st["fy"] = np.full((B, M), 1e5, np.float32)
    st["fr"] = np.full((B, M), 100.0, np.float32)
    for k in ("logn", "gx", "gy"):
        st[k] = np.zeros_like(st[k])
    return st


def fresh(arm, preset, B=1, seed=0):
    rng = np.random.default_rng(seed)
    return arm.reset(preset, rng.random((B, reset_uniform_count(preset))).astype(np.float32))


def place(st, i, x, y, heading, speed=100.0, b=0):
    """Put ship i of env b at (x, y) flying straight along `heading` in empty space."""
    st["x"][b, i], st["y"][b, i] = x, y
    st["c"][b, i], st["s"][b, i] = math.cos(heading), math.sin(heading)
    st["w"][b, i], st["alpha"][b, i], st["ca"][b, i], st["sa"][b, i], st["r"][b, i] = speed, 0.0, 1.0, 0.0, 0.0
    st["logn"][b, i] = st["gx"][b, i] = st["gy"][b, i] = 0.0


def actions(B, N, power=0, stick=0, shoot=0):
    a = np.zeros((B, N, 3), np.int32)
    a[..., 0], a[..., 1], a[..., 2] = power, stick, shoot
    return a


def zeros_noise(preset, B):
    p = PRESETS[preset]
    return (np.zeros((B, p["num_ships"], 2), np.float32),
            np.zeros((B, reset_uniform_count(preset)), np.float32))


def step(arm, preset, st, act):
    spread, uni = zeros_noise(preset, st["x"].shape[0])
    return arm.step(preset, st, act, spread, uni)


def aspect(theta):
    return 1.0 - 0.9 * math.exp(-4.0 * theta * theta / math.pi)


__all__ = ["FLIGHT", "PRESETS", "H", "compare", "compare_dict", "far_fields", "fresh", "place", "actions",
           "zeros_noise", "step", "aspect"]
