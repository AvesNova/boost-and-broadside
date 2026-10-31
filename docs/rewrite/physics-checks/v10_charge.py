"""Smoothstep power curve with the coast charge scaled to the spare power, and static thrust
scaled with power: fill behaviour and performance at a fixed store."""
from v10_common import *
sm = dataclasses.replace(CURVES["smooth"], e_mid=True)
VARIANTS = {"fixed charge": sm,
            "scaled charge": dataclasses.replace(sm, charge_scale=True),
            "scaled charge + static": dataclasses.replace(sm, charge_scale=True, static_scale=True)}
print("Fill from empty, coasting straight:")
for n, sh in VARIANTS.items():
    u, E, _ = run(sh, [[0, 0, 0]] * 60, 104.0, 0.0, 30)
    t = [np.argmax(E >= f * sh.e_max) / 30 if (E >= f * sh.e_max).any() else float("nan") for f in (0.5, 0.75, 0.9, 0.99)]
    print(f"- {n}: 50 / 75 / 90 / 99% at {' / '.join(f'{x:.0f}' for x in t)} s; speed min {u[60:].min():.1f}, "
          f"max {u[60:].max():.1f} m/s; store after 120 s {E[-1] / sh.e_max:.3f}")
print("\\nAt a fixed store (no charging): top speed, coast datum rate at 0.4 / 0.75 / 1.0 stick, best climb:")
def climb(sh, E):
    us = np.linspace(25, 160, 1351)
    P = float(F.engine_power(sh, E)); fs = sh.f_static * ((P / sh.power) ** (2 / 3) if sh.static_scale else 1.0)
    T = P / us / (1 + (P / us / fs) ** 4) ** 0.25
    ex = (T - sh.cd0 * us * us - sh.k_ind * sh.g ** 2 / us ** 2) * us
    return ex.max() / sh.g, us[ex.argmax()]
for n, sh in list(VARIANTS.items())[1:]:
    print(f"- {n}:")
    for f in (0.0, 0.75, 1.0):
        s0 = dataclasses.replace(sh, a0=0.0); E0 = f * sh.e_max
        top = tail(fly(s0, 110.0, E0, [0, 0, 0], 30)[0])
        rates = [math.degrees(tail(fly(s0, 60.0, E0, [0, k, 0], 30)[1])) for k in (2, 4, 6)]
        c, cu = climb(sh, E0)
        print(f"  - store {f:.0%}: top {top:.0f} m/s; turn {' / '.join(f'{r:.0f}' for r in rates)} deg/s; "
              f"best climb {c:.1f} m/s at {cu:.0f} m/s")
