"""Instantaneous and sustained level turn rate vs speed: our model (v8 candidates) and a P-51D
energy-maneuverability model at sea level (point mass, same physics family), in game units.

P-51D (approximate, sea level): m 4,500 kg, S 21.8 m^2, C_Lmax 1.5, C_D0 0.0163, k 0.068
(AR 5.84, e 0.8), shaft 1,490 hp x 80% prop, static thrust ~3 m/s^2 x m (assumption),
+8 g. Converted with speed / 1.485, rates x 4.087.
"""
import math
import numpy as np
LV, LT = 1.4848, 4.087
g_r, rho = 9.81, 1.225
m, S, CLm, CD0, k, P, n_max = 4500.0, 21.8, 1.5, 0.0163, 0.068, 1490 * 745.7 * 0.8, 8.0
T_static = 3.0 * m
def p51(V):
    q = 0.5 * rho * V * V
    n_lift = q * S * CLm / (m * g_r)
    n_inst = min(n_lift, n_max)
    inst = g_r * math.sqrt(max(n_inst ** 2 - 1, 0)) / V
    T = min(P / V, T_static)
    # sustained: T = D0 + Di, Di = k (n W)^2 / (q S)
    n_sq = (T - q * S * CD0) * q * S / (k * (m * g_r) ** 2)
    n_s = min(math.sqrt(max(n_sq, 0)), n_inst)
    sus = g_r * math.sqrt(max(n_s ** 2 - 1, 0)) / V
    return inst, sus
# ours (game units), v8 with p 16, 8 g and the v7-era p 4 for reference
g, cl, kind, power, fs, a0, uc = 27.0, 0.027, 3.8, 1200.0, 20.0, 3.0, 105.0
cd0 = (min(power / uc, fs) - kind * (g / uc ** 2) ** 2 * uc ** 2 - a0) / uc ** 2
def ours(u, a_max, p):
    L = cl * u * u / (1 + (cl * u * u / a_max) ** p) ** (1 / p)
    inst = math.sqrt(max(L * L - g * g, 0)) / u
    x = (power / u) / fs; T = (power / u) / (1 + x ** 4) ** 0.25
    lat_sq = (T - cd0 * u * u) * u * u / kind - g * g          # level turn, no exchange (a0 off)
    lat = min(math.sqrt(max(lat_sq, 0)), math.sqrt(max(L * L - g * g, 0)))
    return inst, lat / u
print(f"{'game u':>6} | {'P-51 inst':>9} {'sust':>5} | {'ours p16 8g inst':>16} {'sust':>5} | {'ours p4 8g inst':>15} {'sust':>5}   (deg/s, game time)")
for u in (35, 45, 55, 65, 75, 85, 90, 95, 105, 115, 125, 135):
    pi, ps = p51(u * LV)
    oi, os_ = ours(u, 216.0, 16)
    qi, qs = ours(u, 216.0, 4)
    print(f"{u:>6} | {math.degrees(pi) * LT:9.0f} {math.degrees(ps) * LT:5.0f} | {math.degrees(oi):16.0f} {math.degrees(os_):5.0f} | {math.degrees(qi):15.0f} {math.degrees(qs):5.0f}")
# P-51 corner and stall in game units
Vc = math.sqrt(2 * n_max * m * g_r / (rho * S * CLm)); Vs = math.sqrt(2 * m * g_r / (rho * S * CLm))
print(f"\nP-51 model: stall {Vs / LV:.1f}, corner {Vc / LV:.1f} m/s game (real {Vs:.0f}, {Vc:.0f})")
# where is the P-51's best sustained turn and its value
best = max(((p51(v / 1.0)[1], v) for v in np.linspace(40, 200, 400)))
print(f"P-51 model best sustained: {math.degrees(best[0]) * LT:.0f} deg/s game at {best[1] / LV:.0f} m/s game (real {math.degrees(best[0]):.1f} deg/s at {best[1]:.0f} m/s)")
best_o = max(((ours(u, 216.0, 16)[1], u) for u in np.linspace(32, 120, 400)))
print(f"ours best sustained: {math.degrees(best_o[0]):.0f} deg/s at {best_o[1]:.0f} m/s")
