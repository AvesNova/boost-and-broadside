"""Sustained turn in several energy conditions, and the effect of the induced-drag factor
(v8: p 16, nose x2, 8 g; 1200 W, 20 N, coast climb 3 m/s^2; cd0 re-tuned for a 105 m/s cruise)."""
import os, sys, math, dataclasses
sys.path.insert(0, os.path.dirname(__file__))
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy as np
import flight8 as F
NOF = tuple(jnp.zeros((0,)) for _ in range(5)); HZ = 30
def make(k_mult=1.0, power=1200.0, fs=20.0, a0=3.0, u_c=105.0):
    b = F.Ship(); kind = b.k_ind * k_mult
    ind = kind * (b.g / u_c ** 2) ** 2 * u_c ** 2
    return dataclasses.replace(b, k_ind=kind, power=power, f_static=fs, a0=a0, cd0=(min(power / u_c, fs) - ind - a0) / u_c ** 2, soft_k=4, nose_mult=2.0)
def fly(sh, speed, E, act, sec):
    st = F.init(jnp.array([0.0]), jnp.array([0.0]), jnp.array([0.0]), jnp.array([float(speed)]), jnp.array([float(E)]), NOF)
    a = jnp.array([act])
    def body(s, i):
        s2 = F.step(sh, NOF, s, a, 1 / HZ)
        return s2, (s2[4][0], jnp.arctan2(s2[3][0], s2[2][0]) - s2[5][0], s2[9][0])
    u, th, e = jax.jit(lambda s: jax.lax.scan(body, s, None, length=int(sec * HZ)))(st)[1]
    th = np.unwrap(np.asarray(th))
    return np.asarray(u), np.diff(th, prepend=th[0]) * HZ, np.asarray(e)
def sustained(sh, E, act, sec=25.0):
    u, r, e = fly(sh, 90.0, E, act, sec)
    us, w = float(np.mean(u[-2 * HZ:])), float(np.mean(r[-2 * HZ:]))
    return us, math.degrees(w), us / max(abs(w), 1e-9), float((e[-1] - e[-HZ]))
def excess_curve(sh):
    us = np.linspace(32.0, 140.0, 1000)
    x = (sh.power / us) / sh.f_static; T = (sh.power / us) / (1 + x ** 4) ** 0.25
    D = (sh.cd0 + sh.k_ind * (sh.g / us ** 2) ** 2) * us ** 2
    pe = (T - D) * us; i = int(np.argmax(pe))
    top = us[np.argmax(pe < 0)] if (pe < 0).any() else float("nan")
    return pe[i], us[i], top

sh = make()
inf = dataclasses.replace(sh, e_max=1e9)
print("Sustained full-stick turn (speed m/s, rate deg/s, radius m, store W), k_ind x1:")
for name, s_, E, act in (("coast, store mid (charging at a0)", sh, 5000.0, [0, 6, 0]),
                         ("coast, store full (no charging)", sh, sh.e_max, [0, 6, 0]),
                         ("boost (dive), unlimited store", inf, 5e8, [1, 6, 0]),
                         ("boost (dive), store empty (floor)", sh, 0.0, [1, 6, 0]),
                         ("coast, engine off, unlimited store", dataclasses.replace(inf, power=0.0, f_static=1e-6), 5e8, [0, 6, 0]),
                         ("boost, engine off, unlimited store", dataclasses.replace(inf, power=0.0, f_static=1e-6), 5e8, [1, 6, 0])):
    us, w, rad, dE = sustained(s_, E, act)
    print(f"  {name:<38} {us:6.1f} m/s {w:6.1f} deg/s  r {rad:5.0f} m  store {dE:+7.0f} W")
print("\nP-51 model: best sustained 70 deg/s at 49 m/s (game units)\n")
print(f"{'k_ind x':>7} {'cd0':>8} | {'coast full-stick: m/s deg/s r':>30} | {'best sust (store full) deg/s @ m/s':>34} | {'max excess W @ m/s':>18} {'climb m/s':>9} {'top m/s':>7} | {'inst peak':>9} {'speed lost in 1 s full stick':>28}")
for km in (1.0, 1.5, 2.0, 3.0, 4.0):
    s_ = make(km)
    us, w, rad, _ = sustained(s_, s_.e_max, [0, 6, 0])
    best = max((sustained(s_, s_.e_max, [0, st, 0], 20.0)[1], sustained(s_, s_.e_max, [0, st, 0], 20.0)[0]) for st in (2, 4, 6))
    pe, ue, top = excess_curve(s_)
    u, r, e = fly(s_, 95.0, 5000.0, [0, 6, 0], 1.0)
    print(f"{km:>7.1f} {s_.cd0:8.2e} | {us:9.1f} {w:7.1f} {rad:6.0f}       | {best[0]:20.1f} @ {best[1]:5.0f}       | {pe:9.0f} @ {ue:4.0f} {pe / s_.g:9.1f} {top:7.0f} | {math.degrees(np.max(np.abs(r))):9.0f} {95 - u[-1]:28.0f}")
