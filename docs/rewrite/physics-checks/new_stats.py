"""Headline stats of the v7 model (1200 W, 20 N, coast climb 3 m/s^2), 30 Hz tick, no fields."""
import os, sys, math, dataclasses
sys.path.insert(0, os.path.dirname(__file__))
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy as np
exec(open(os.path.join(os.path.dirname(__file__), "v7_sweep.py")).read().split("CONFIGS =")[0])
sh, _ = make(1200.0, 20.0, 3.0)
def fl(speed, E, act, sec):
    u, r, e = fly(sh, speed, E, act, sec); return np.asarray(u), np.asarray(r), np.asarray(e)
def steady(x, sec=3): return float(np.mean(x[-int(sec * HZ):]))
print(f"stall (min level) {math.sqrt(sh.g / sh.cl_max):.1f} m/s; corner {sh.u_corner:.0f} m/s; instantaneous max rate at corner "
      f"{math.degrees(math.sqrt((sh.a_max / 2**0.25)**2 - sh.g**2) / sh.u_corner):.0f} deg/s; min instantaneous radius 1/C_Lmax {1 / sh.cl_max:.0f} m")
u, r, e = fl(105.0, 0.0, [0, 0, 0], 30.0); print(f"coast straight while charging: {steady(u):.1f} m/s")
u, r, e = fl(105.0, sh.e_max, [0, 0, 0], 60.0); print(f"coast straight, store full (top level speed): {steady(u):.1f} m/s")
u, r, e = fl(105.0, sh.e_max, [1, 0, 0], 3.0); print(f"boost straight from a full store: peak {u.max():.0f} m/s")
for name, act in (("coast 1/3", [0, 2, 0]), ("coast 2/3", [0, 4, 0]), ("coast full", [0, 6, 0])):
    u, r, e = fl(100.0, sh.e_max / 2, act, 30.0)
    w = steady(r); us = steady(u)
    print(f"{name}: sustained {us:.1f} m/s, {math.degrees(w):.0f} deg/s, radius {us / max(abs(w), 1e-9):.0f} m, store {float(e[-1] - e[-HZ]):+.0f} W")
u, r, e = fl(sh.u_corner, sh.e_max, [1, 6, 0], 10.0)
alive = e > 0.05 * sh.e_max
idx = np.where(alive)[0]; win = idx[len(idx) // 3:] if len(idx) > 6 else idx
print(f"boost + full stick (diving turn) from a full store at corner speed: store lasts {len(idx) / HZ:.1f} s; "
      f"during it {np.mean(u[win]):.0f} m/s, {math.degrees(np.mean(r[win])):.0f} deg/s, radius {np.mean(u[win]) / np.mean(r[win]):.0f} m")
for u0 in (40, 60, sh.u_corner, 140):
    u, r, e = fl(float(u0), sh.e_max / 2, [0, 6, 0], 0.8)
    pk = np.max(np.abs(r)); t90 = int(np.argmax(np.abs(r) >= 0.9 * pk)) / HZ
    print(f"  from {u0:.0f} m/s, full stick: peak rate {math.degrees(pk):.0f} deg/s, time to 90% {t90:.2f} s")
