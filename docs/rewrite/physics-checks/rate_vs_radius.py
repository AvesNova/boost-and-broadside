"""Where do max turn rate and min turn radius occur (speed and lift coefficient)?
Level turns, per unit mass, game units. Lateral = sqrt((C_L u^2)^2 - g^2) capped by the g limit.
Instantaneous: limited by lift only. Sustained: thrust = drag (coasting, no climb)."""
import math, numpy as np
LV, LT, LL = 1.4848, 4.087, 6.068
def analyse(name, g, clmax, amax, cd0, k, thrust):
    us = np.linspace(g ** 0.5 / clmax ** 0.5 * 1.001, 160, 4000)
    best = {}
    for kind in ("instant", "sustained"):
        rate_b = rad_b = None
        for u in us:
            L = min(clmax * u * u, amax)
            lat_max = math.sqrt(max(L * L - g * g, 0))
            if kind == "sustained":
                lat_s = (thrust(u) - cd0 * u * u) * u * u / k - g * g
                lat = min(math.sqrt(max(lat_s, 0)), lat_max)
            else:
                lat = lat_max
            if lat <= 0: continue
            cl = math.sqrt(lat * lat + g * g) / (u * u) / clmax
            rate, rad = lat / u, u * u / lat
            if rate_b is None or rate > rate_b[0]: rate_b = (rate, u, cl, rad)
            if rad_b is None or rad < rad_b[0]: rad_b = (rad, u, cl, rate)
        best[kind] = (rate_b, rad_b)
    print(f"\n{name}")
    for kind, (rb, db) in best.items():
        print(f"  {kind:9}: max rate {math.degrees(rb[0]):5.1f} deg/s at {rb[1]:5.1f} m/s, stick (C_L/C_Lmax) {rb[2]:.2f}, radius {rb[3]:.0f} m")
        print(f"  {'':9}  min radius {db[0]:5.1f} m at {db[1]:5.1f} m/s, stick {db[2]:.2f}, rate {math.degrees(db[3]):.1f} deg/s")
# P-51D, sea level, converted to game units (per unit mass: divide forces by m; scale speeds/LV, accel*LT/LV)
m, S, rho, CLm, CD0, kk, Pw = 4500.0, 21.8, 1.225, 1.5, 0.0163, 0.068, 1490 * 745.7 * 0.8
LA = LV / LT
g_r = 9.81
clmax_g = rho * S * CLm / (2 * m) * LV * LV / LA         # per-metre lift coefficient in game units
cd0_g = rho * S * CD0 / (2 * m) * LV * LV / LA
k_g = kk * 2 * m / (rho * S) * LA / (LV * LV)           # induced factor in our per-metre form
def thrust_p51(u):
    V = u * LV; return min(Pw / V, 3.0 * m) / m / LA
analyse("P-51D (energy-maneuverability model, game units)", g_r / LA, clmax_g, 8 * g_r / LA, cd0_g, k_g, thrust_p51)
def thrust_ours(u):
    x = (1200.0 / u) / 20.0; return (1200.0 / u) / (1 + x ** 4) ** 0.25
k_ours = 3 * 3.8; cd0_ours = (min(1200 / 105, 20) - k_ours * (27 / 105 ** 2) ** 2 * 105 ** 2 - 3.0) / 105 ** 2
analyse("Ours (final config, coasting with no climb)", 27.0, 0.027, 216.0, cd0_ours, k_ours, thrust_ours)
