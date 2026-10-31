"""Fields (SPEC 4), flight (SPEC 3) and bullet motion (SPEC 6.2) against the reference."""

import math

import numpy as np
import pytest

from conftest import golden
from helpers import FLIGHT, H, compare, compare_dict, first_move, half_drag

SHIP_KEYS = ("x", "y", "c", "s", "w", "alpha", "ca", "sa", "r", "E", "logn", "gx", "gy")


# ------------------------------------------------------------------ fields

def test_fields_match_reference(arm):
    g = golden("fields.npz")
    logn, gx, gy = arm.fields(g["fields"], g["points"]["x"], g["points"]["y"])
    for key, val in (("logn", logn), ("gx", gx), ("gy", gy)):
        compare(val, g["expected"][key], key, atol=2e-6, rtol=1e-5)


def test_field_gradient_is_the_gradient_of_the_index(arm):
    g = golden("fields.npz")
    x, y = g["points"]["x"].astype(np.float64), g["points"]["y"].astype(np.float64)
    step = 0.25
    f = g["fields"]
    lp, _, _ = arm.fields(f, (x + step).astype(np.float32), y.astype(np.float32))
    lm, _, _ = arm.fields(f, (x - step).astype(np.float32), y.astype(np.float32))
    _, gx, _ = arm.fields(f, x.astype(np.float32), y.astype(np.float32))
    numeric = (lp.astype(np.float64) - lm) / (2 * step)
    compare(gx, numeric, "gx", atol=3e-4, rtol=2e-2, max_bad=0.01, context="finite difference: ")


def test_index_is_bounded_by_the_levels(arm):
    g = golden("fields.npz")
    logn, _, _ = arm.fields(g["fields"], g["points"]["x"], g["points"]["y"])
    assert np.all(np.abs(logn) <= 2 * 0.5 * math.log(2.0) + 1e-6)


# ------------------------------------------------------------------ flight

def test_flight_tick_matches_reference(arm):
    g = golden("flight_tick.npz")
    out = arm.flight_tick(g["ship"], g["power"], g["stick"], g["fields"])
    compare_dict({k: out[k] for k in SHIP_KEYS}, g["expected"], rtol=2e-5, max_bad=1e-3, context="flight tick: ")


def _fly(arm, start, power, stick, ticks, flight=None, fields=None):
    if fields is None:
        fields = dict(fx=np.zeros((1, 1), np.float32) + 1e5, fy=np.zeros((1, 1), np.float32) + 1e5,
                      fr=np.full((1, 1), 100.0, np.float32), fw=np.full((1, 1), 40.0, np.float32),
                      flevel=np.ones((1, 1), np.int32))
    st = {k: v.copy() for k, v in start.items()}
    hist = []
    for _ in range(ticks):
        st = dict(arm.flight_tick(st, power, stick, fields, flight))
        if "ship_type" in start:  # flight_tick returns the 13 flight keys; the type is fixed
            st["ship_type"] = start["ship_type"]
        hist.append({k: np.asarray(v).copy() for k, v in st.items()})
    return hist


def test_flight_trajectory_first_four_seconds(arm):
    g = golden("flight_trajectory.npz")
    hist = _fly(arm, g["start"], g["power"], g["stick"], 120)
    for key in ("x", "y", "w", "E", "alpha"):
        got = np.stack([h[key][0] for h in hist])
        atol = dict(x=0.05, y=0.05, w=0.01, E=0.5, alpha=1e-4)[key]
        compare(got, g[f"traj_{key}"], key, atol=atol, rtol=1e-4, context="trajectory: ")


def test_flight_maneuvers(arm):
    """Steady-state handling numbers (the flight envelope) within 0.5%."""
    spec = golden("flight_maneuvers.json")
    man = spec["maneuvers"]
    n = len(man)
    z = np.zeros((1, n), np.float32)
    start = dict(x=z.copy(), y=z.copy(), c=z + 1.0, s=z.copy(), w=np.array([[m["speed"] for m in man]], np.float32),
                 alpha=z.copy(), ca=z + 1.0, sa=z.copy(), r=z.copy(),
                 E=np.array([[m["store"] for m in man]], np.float32), logn=z.copy(), gx=z.copy(), gy=z.copy(),
                 ship_type=np.array([[m.get("ship_type", 0) for m in man]], np.int32))
    power = np.array([[m["power"] for m in man]], np.int32)
    stick = np.array([[m["stick"] for m in man]], np.int32)
    hist = _fly(arm, start, power, stick, max(m["seconds"] for m in man) * 30)
    w = np.stack([h["w"][0] for h in hist]).astype(np.float64)
    E = np.stack([h["E"][0] for h in hist]).astype(np.float64)
    path = np.unwrap(np.stack([np.arctan2(h["s"][0], h["c"][0]) - h["alpha"][0] for h in hist]).astype(np.float64),
                     axis=0)
    rate = np.degrees(np.diff(path, axis=0, prepend=path[:1]) * 30.0)
    errors = []
    for i, m in enumerate(man):
        want = spec["stats"][m["name"]]
        k = want["ticks"]
        got = dict(final_speed=w[k - 30:k, i].mean(), peak_speed=w[:k, i].max(), final_store=E[k - 1, i],
                   final_turn_rate_deg=rate[k - 30:k, i].mean(), peak_turn_rate_deg=np.abs(rate[:k, i]).max(),
                   speed_after_1s=w[29, i])
        for key, val in got.items():
            tol = max(0.005 * abs(want[key]), dict(final_store=5.0).get(key, 0.05))
            if abs(val - want[key]) > tol:
                errors.append(f"{m['name']}.{key}: got {val:.3f}, want {want[key]:.3f}")
    assert not errors, "; ".join(errors)


@pytest.mark.parametrize("with_fields", [False, True])
def test_energy_is_conserved_without_drag_or_engine(arm, with_fields):
    """With drag, engine and induced drag off, 1/2 w^2 + E is constant (exchange is lossless)."""
    rng = np.random.default_rng(1)
    N = 256
    off = dict(cd0=0.0, k_ind=0.0, power=0.0, p_min=0.0, f_static=1e-6)
    fields = None
    if with_fields:
        fields = dict(fx=rng.uniform(-600, 600, (1, 8)).astype(np.float32),
                      fy=rng.uniform(-600, 600, (1, 8)).astype(np.float32),
                      fr=rng.uniform(100, 300, (1, 8)).astype(np.float32), fw=np.full((1, 8), 40.0, np.float32),
                      flevel=rng.choice([-2, -1, 1, 2], (1, 8)).astype(np.int32))
    x = rng.uniform(-500, 500, (1, N)).astype(np.float32)
    y = rng.uniform(-500, 500, (1, N)).astype(np.float32)
    if fields is not None:
        logn, gx, gy = arm.fields(fields, x, y)
    else:
        logn = gx = gy = np.zeros((1, N), np.float32)
    hd = rng.uniform(0, 2 * math.pi, (1, N))
    start = dict(x=x, y=y, c=np.cos(hd).astype(np.float32), s=np.sin(hd).astype(np.float32),
                 w=rng.uniform(60, 160, (1, N)).astype(np.float32), alpha=np.zeros((1, N), np.float32),
                 ca=np.ones((1, N), np.float32), sa=np.zeros((1, N), np.float32),
                 r=np.zeros((1, N), np.float32), E=rng.uniform(1000, 9000, (1, N)).astype(np.float32),
                 logn=np.asarray(logn, np.float32), gx=np.asarray(gx, np.float32), gy=np.asarray(gy, np.float32))
    e0 = 0.5 * start["w"].astype(np.float64) ** 2 + start["E"]
    power = rng.integers(0, 3, (1, N)).astype(np.int32)
    stick = rng.integers(0, 7, (1, N)).astype(np.int32)
    hist = _fly(arm, start, power, stick, 60, flight=off, fields=fields)
    end = hist[-1]
    e1 = 0.5 * end["w"].astype(np.float64) ** 2 + end["E"]
    keep = (end["E"] > 300) & (end["E"] < FLIGHT["e_max"] - 300)  # away from the store-end fades
    rel = np.abs(e1 - e0)[keep] / e0[keep]
    assert keep.sum() > N // 4
    assert rel.max() < 1e-4, f"energy drift {rel.max():.2e}"


# ------------------------------------------------------------------ bullets

def test_bullet_tick_matches_reference(arm):
    g = golden("bullet_tick.npz")
    out = arm.bullet_tick(g["bullets"], g["fields"])
    compare_dict(out, g["expected"], rtol=2e-5, max_bad=1e-3, context="bullet tick: ")


def test_bullets_fly_straight_in_empty_space(arm):
    rng = np.random.default_rng(2)
    P = 64
    ang = rng.uniform(0, 2 * math.pi, (1, P))
    b = dict(bx=rng.uniform(-100, 100, (1, P)).astype(np.float32), by=rng.uniform(-100, 100, (1, P)).astype(np.float32),
             bdx=np.cos(ang).astype(np.float32), bdy=np.sin(ang).astype(np.float32),
             bw=rng.uniform(300, 700, (1, P)).astype(np.float32), blogn=np.zeros((1, P), np.float32),
             bgx=np.zeros((1, P), np.float32), bgy=np.zeros((1, P), np.float32))
    fields = dict(fx=np.full((1, 1), 1e5, np.float32), fy=np.full((1, 1), 1e5, np.float32),
                  fr=np.full((1, 1), 100.0, np.float32), fw=np.full((1, 1), 40.0, np.float32),
                  flevel=np.ones((1, 1), np.int32))
    out = arm.bullet_tick(b, fields)
    w = b["bw"].astype(np.float64)
    step_len = np.array([[first_move(v) for v in row] for row in w])
    compare(out["bx"], b["bx"] + b["bdx"] * step_len, "bx", atol=1e-3, rtol=1e-6)
    compare(out["by"], b["by"] + b["bdy"] * step_len, "by", atol=1e-3, rtol=1e-6)
    compare(out["bw"], half_drag(half_drag(w, 0.0), 0.0), "bw", atol=1e-3, rtol=1e-6)


def test_drag_slows_proper_speed_and_direction_stays_unit(arm):
    """Proper speed changes only by drag: half a tick with the start index, half with the new."""
    g = golden("bullet_tick.npz")
    b = g["bullets"]
    out = arm.bullet_tick(b, g["fields"])
    want = half_drag(half_drag(b["bw"].astype(np.float64), b["blogn"]), out["blogn"].astype(np.float64))
    compare(out["bw"], want, "bw", atol=1e-3, rtol=1e-6)
    norm = np.hypot(out["bdx"].astype(np.float64), out["bdy"])
    compare(norm, np.ones_like(norm), "|d|", atol=1e-5, rtol=0.0)
