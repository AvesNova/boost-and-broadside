import os, sys, math, dataclasses
sys.path.insert(0, os.path.dirname(__file__))
exec(open(os.path.join(os.path.dirname(__file__), "v7_sweep.py")).read().split("CONFIGS =")[0])
import numpy as np
def max_excess(sh):
    u = np.linspace(32.0, 160.0, 2000)
    T = np.minimum(sh.power / u, sh.f_static) / (1 + (np.minimum(sh.power / u, sh.f_static) / np.maximum(sh.power / u, sh.f_static)) ** 4) ** 0.25 if False else None
    x = (sh.power / u) / sh.f_static
    T = (sh.power / u) / (1 + x ** 4) ** 0.25
    D = (sh.cd0 + sh.k_ind * (sh.g / u ** 2) ** 2) * u ** 2
    P = (T - D) * u
    i = int(np.argmax(P))
    return P[i], u[i]
for power, fs, a0 in ((800, 15, 2.0), (1200, 20, 3.0), (1600, 20, 3.0)):
    sh, charge = make(power, fs, a0)
    pe, ue = max_excess(sh)
    sh = dataclasses.replace(sh, bullet_cost=1.2 * pe / HZ)
    print(f"engine {power} W / {fs} N: max excess power {pe:.0f} W at {ue:.0f} m/s; coast charge {charge:.0f} W -> bullet {sh.bullet_cost:.1f} J ({sh.bullet_cost * HZ:.0f} W of fire)")
    for name, act in (("coasting", [0, 0, 1]), ("climbing", [2, 0, 1]), ("diving", [1, 0, 1]), ("hard turn", [0, 6, 1])):
        u, r, E = fly(sh, 105.0, sh.e_max, act, 300.0)
        t = t_reach(E, sh.bullet_cost, above=False)
        print(f"  fire while {name:>9}: empty after {t:6.1f} s ({t * HZ:.0f} bullets)" if t == t else f"  fire while {name:>9}: never empty in 300 s (store {float(E[-1]):.0f} J)")
