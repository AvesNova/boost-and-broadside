"""Sweep v6 settings. Drag is set so coasting (neutral pitch) cruises at 105 m/s while
charging; static thrust caps at P / 40 m/s. Bullet cost = 1.5 x the neutral charge power
at cruise, spread over 30 bullets per second."""
import os, sys, math, dataclasses
sys.path.insert(0, os.path.dirname(__file__))
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import flight6 as F
LV, LT = 1.4848, 4.087
NOF = tuple(jnp.zeros((0,)) for _ in range(5))
HZ = 30   # physics tick = decision tick: one bullet per step = 30 bullets per second

def make(power, gamma0_deg, e_max, gamma_max_deg=30.0, u_c=105.0):
    base = F.Ship()
    g0 = math.radians(gamma0_deg)
    ind = base.k_ind * (base.g * math.cos(g0) / u_c ** 2) ** 2 * u_c ** 2
    cd0 = (power / u_c - ind - base.g * math.sin(g0)) / u_c ** 2
    sh = dataclasses.replace(base, power=power, cd0=cd0, f_static=power / 40.0, e_max=e_max,
                             gamma0=g0, gamma_max=math.radians(gamma_max_deg))
    charge = base.mass * base.g * u_c * math.sin(g0)
    return dataclasses.replace(sh, bullet_cost=1.5 * charge / 30.0), charge

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

CONFIGS = [(400, 3, 10_000), (800, 3, 10_000), (1200, 3, 10_000), (1200, 6, 10_000), (1600, 5, 10_000), (1600, 5, 6_000)]
for power, g0, emax in CONFIGS:
    sh, charge = make(power, g0, emax)
    print(f"\n=== engine {power} W, neutral pitch {g0} deg, store {emax / 1000:.0f} kJ  (cd0 {sh.cd0:.2e}, static thrust {sh.f_static:.0f} N)")
    u, r, E = fly(sh, 105.0, 0.0, [0, 0, 0], 120.0)
    tf = t_reach(E, 0.98 * emax)
    print(f"  coast from empty: neutral charge {charge:.0f} W; full after {tf:.1f} s ({tf * LT:.0f} s real); top level speed once full {float(u[-1]):.0f} m/s")
    u, r, E = fly(sh, 105.0, 0.0, [2, 0, 0], 60.0)
    tf = t_reach(E, 0.98 * emax); i = min(int(tf * HZ) if tf == tf else -1, len(u) - 1)
    print(f"  climb (reverse) from empty: full after {tf:.1f} s ({tf * LT:.0f} s real) at ~{float(u[i // 2]):.0f} m/s")
    u, r, E = fly(sh, 105.0, emax, [1, 0, 0], 60.0)
    print(f"  dive (boost) from full: peak {float(u.max()):.0f} m/s (real {float(u.max()) * LV:.0f}); empty after {t_reach(E, 0.02 * emax, above=False):.1f} s")
    u, r, E = fly(sh, 100.0, emax / 2, [0, 6, 0], 10.0)
    print(f"  coast full stick: {float(u[-1]):.0f} m/s, {math.degrees(float(r[-1])):.0f} deg/s ({math.degrees(float(r[-1])) / LT:.1f} real), store {float((E[-1] - E[-HZ]) ):+.0f} W")
    u, r, E = fly(sh, 25.0, emax / 2, [0, 0, 0], 10.0)
    print(f"  stall from 25 m/s: 40 m/s after {t_reach(u, 40.0):.2f} s")
    u, r, E = fly(sh, 105.0, emax, [0, 0, 1], 60.0)
    print(f"  bullet cost {sh.bullet_cost:.1f} J (fire drains {sh.bullet_cost * 30:.0f} W): continuous fire while coasting empties a full store in {t_reach(E, sh.bullet_cost, above=False):.1f} s")
    u, r, E = fly(sh, 105.0, emax, [2, 0, 1], 60.0)
    print(f"  ... while climbing (reverse): store after 20 s {float(E[20 * HZ]):.0f} J")
