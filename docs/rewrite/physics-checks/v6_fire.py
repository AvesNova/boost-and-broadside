import os, sys, math, dataclasses
sys.path.insert(0, os.path.dirname(__file__))
exec(open(os.path.join(os.path.dirname(__file__), "v6_sweep.py")).read().split("CONFIGS =")[0])
for power, g0, emax in ((1600, 5, 10_000), (1200, 6, 10_000)):
    sh, charge = make(power, g0, emax)
    u, r, E = fly(sh, 105.0, 0.0, [2, 0, 0], 60.0)
    best = float(jnp.max((E[HZ:] - E[:-HZ])))                    # best one-second charge while climbing
    sh = dataclasses.replace(sh, bullet_cost=1.5 * best / 30.0)
    print(f"engine {power} W, neutral {g0} deg: best climb charge {best:.0f} W -> bullet cost {sh.bullet_cost:.0f} J ({sh.bullet_cost * 30:.0f} W of fire)")
    for name, act in (("coasting", [0, 0, 1]), ("climbing", [2, 0, 1]), ("diving", [1, 0, 1]), ("hard turn", [0, 6, 1])):
        u, r, E = fly(sh, 105.0, emax, act, 60.0)
        t = t_reach(E, sh.bullet_cost, above=False)
        print(f"  continuous fire while {name:>9}: store empty after {t:5.1f} s ({t * 30:.0f} bullets), speed then {float(u[int(t * HZ)]) if t == t else float('nan'):.0f} m/s")
