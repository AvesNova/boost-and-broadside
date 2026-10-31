"""Sweep v7. Drag set so coasting cruises at 105 m/s. Static thrust F_s < a_climb so
reverse always slows. Bullet cost = 1.5 x the coast charge at cruise / 30."""
import os, sys, math, dataclasses
sys.path.insert(0, os.path.dirname(__file__))
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import flight7 as F
LV, LT = 1.4848, 4.087
NOF = tuple(jnp.zeros((0,)) for _ in range(5))
HZ = 30

def make(power, f_static, a0, e_max=10_000.0, a_climb=27.0, a_dive=27.0, u_c=105.0):
    base = F.Ship()
    ind = base.k_ind * (base.g / u_c ** 2) ** 2 * u_c ** 2
    cd0 = (min(power / u_c, f_static) - ind - a0) / u_c ** 2
    sh = dataclasses.replace(base, power=power, cd0=cd0, f_static=f_static, e_max=e_max, a0=a0, a_climb=a_climb, a_dive=a_dive)
    charge = base.mass * u_c * a0
    return dataclasses.replace(sh, bullet_cost=1.5 * charge / HZ), charge

def fly(sh, speed, E, act, seconds):
    n = int(round(seconds * HZ))
    st = F.init(jnp.array([0.0]), jnp.array([0.0]), jnp.array([0.0]), jnp.array([float(speed)]), jnp.array([float(E)]), NOF)
    a = jnp.array([act])
    def body(s, i):
        s2 = F.step(sh, NOF, s, a, 1 / HZ)
        return s2, (s2[4][0], s2[8][0], s2[9][0])
    return jax.jit(lambda s: jax.lax.scan(body, s, None, length=n))(st)[1]

def t_reach(series, value, above=True):
    hit = series >= value if above else series <= value
    i = int(jnp.argmax(hit))
    return i / HZ if bool(hit[i]) else float("nan")

CONFIGS = [(800, 15, 2.0), (1200, 15, 2.0), (1200, 20, 3.0), (1600, 20, 3.0)]
for power, fs, a0 in CONFIGS:
    sh, charge = make(power, fs, a0)
    if sh.cd0 <= 0:
        print(f"skip {power} W / {fs} N: thrust too weak for 105 m/s cruise"); continue
    print(f"\n=== engine {power} W, static thrust {fs} N, coast climb {a0} m/s^2  (cd0 {sh.cd0:.2e}; coast charge {charge:.0f} W; bullet {sh.bullet_cost:.1f} J)")
    u, r, E = fly(sh, 105.0, 0.0, [0, 0, 0], 120.0)
    print(f"  coast from empty: full after {t_reach(E, 0.98 * sh.e_max):.0f} s; then top level speed {float(u[-1]):.0f} m/s")
    u, r, E = fly(sh, 105.0, 0.0, [2, 0, 0], 20.0)
    i = int(jnp.argmax(E))
    print(f"  reverse (climb) from 105 m/s, empty: speed {float(u[HZ]):.0f} m/s after 1 s, {float(u[2 * HZ]):.0f} after 2 s; store peaks {float(E.max()):.0f} J at {float(u[i]):.0f} m/s ({i / HZ:.1f} s)")
    u, r, E = fly(sh, 105.0, sh.e_max, [1, 0, 0], 30.0)
    print(f"  boost (dive) from full: peak {float(u.max()):.0f} m/s; store empty after {t_reach(E, 0.02 * sh.e_max, above=False):.1f} s")
    u, r, E = fly(sh, 100.0, sh.e_max / 2, [0, 6, 0], 10.0)
    print(f"  coast full stick: {float(u[-1]):.0f} m/s, {math.degrees(float(r[-1])):.0f} deg/s, store {float(E[-1] - E[-HZ]):+.0f} W")
    u, r, E = fly(sh, 25.0, sh.e_max / 2, [0, 0, 0], 10.0)
    print(f"  stall 25 -> 40 m/s: {t_reach(u, 40.0):.2f} s")
    for name, act in (("coasting", [0, 0, 1]), ("climbing", [2, 0, 1]), ("diving", [1, 0, 1]), ("hard turn", [0, 6, 1])):
        u, r, E = fly(sh, 105.0, sh.e_max, act, 120.0)
        t = t_reach(E, sh.bullet_cost, above=False)
        print(f"  fire while {name:>9}: empty after {t:6.1f} s ({t * HZ:.0f} bullets)" if t == t else f"  fire while {name:>9}: never empty in 120 s (store {float(E[-1]):.0f} J, speed {float(u[-1]):.0f})")
