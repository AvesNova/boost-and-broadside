"""Check the climb/dive equations against a 3D point mass in a gravity field.

Reference (per unit mass, flight-path angle gamma, heading psi, height h):
  u'     = T(u) cos(alpha) - D - g sin(gamma)
  gamma' = (L_v - g cos(gamma)) / u
  psi'   = L_h / (u cos(gamma))
  x', y' = u cos(gamma) (cos psi, sin psi);  h' = u sin(gamma);  E = g h
A pilot law tracks a commanded climb angle: L_v = g cos(gamma) + u K (gamma_c - gamma), with
the lift vector capped by the same budget L(u). Drag (C_D0 + k C_L^2) u^2 on the total lift.
Ours (memoryless): a_x = g sin(gamma_c) applied at once, horizontal speed u, lift to hold g.
Same engine, drag, budget; no turning; engine thrust taken along the path in both.
"""
from v9_common import *
sh = ship(levels="stick")
g = sh.g
def thrust(u): x = (sh.power / u) / sh.f_static; return (sh.power / u) / (1 + x ** 4) ** 0.25
def budget(u):
    a = sh.cl_max * u * u; x = a / sh.a_max; return a / (1 + x ** 16) ** (1 / 16)
def ref(u0, E0, a_cmd, sec, K=3.0, dt=1e-3):
    gc = math.asin(max(-1.0, min(1.0, a_cmd / g)))
    u, gam, h, xh = u0, 0.0, E0 / g, 0.0
    out = []
    for i in range(int(sec / dt)):
        Lv = g * math.cos(gam) + u * K * (gc - gam)
        L = budget(u)
        Lv = max(-L, min(L, Lv))
        cl = abs(Lv) / max(u * u, 1e-6)
        D = (sh.cd0 + sh.k_ind * cl * cl) * u * u
        du = thrust(u) - D - g * math.sin(gam)
        dg = (Lv - g * math.cos(gam)) / u
        u += du * dt; gam += dg * dt; h += u * math.sin(gam) * dt; xh += u * math.cos(gam) * dt
        if (i + 1) % int(0.5 / dt) == 0: out.append((u, g * h, xh, math.degrees(gam)))
    return out
cases = [("coast (a0 = 3 m/s^2)", 0, sh.a0, 105.0, 5000.0), ("reverse: full climb (a = g)", 2, sh.a_climb, 105.0, 0.0),
         ("half climb (a = g/2)", 2, g / 2, 105.0, 0.0), ("boost: full dive (a = -g)", 1, -sh.a_dive, 60.0, 90000.0)]
for name, pm, a_cmd, u0, E0 in cases:
    print(f"\n{name}: implied climb angle {math.degrees(math.asin(max(-1, min(1, a_cmd / g)))):.0f} deg")
    r = ref(u0, E0, a_cmd, 3.0)
    if pm is not None:
        u, th, e, al = fly(dataclasses.replace(sh, e_max=1e5, a_climb=abs(a_cmd) if pm == 2 else sh.a_climb), u0, E0, [pm, 0, 0], 3.0)
    else:
        u = e = None
    print(f"   t    | ref: speed  energy(gh)  horizontal  angle | ours: speed  energy  horizontal")
    for j, t in enumerate((0.5, 1.0, 1.5, 2.0, 3.0)):
        k = int(t / 0.5) - 1
        ru, re, rx, rg = r[k]
        if u is not None:
            i = int(t * HZ) - 1
            ox = float(np.sum(u[: i + 1]) / HZ)
            print(f"  {t:3.1f} s | {ru:9.1f} {re:10.0f} {rx:11.0f} {rg:6.0f} | {u[i]:10.1f} {e[i]:7.0f} {ox:10.0f}")
        else:
            print(f"  {t:3.1f} s | {ru:9.1f} {re:10.0f} {rx:11.0f} {rg:6.0f} |")
