"""Express the v4 ship in SI and scale it to a P-51D.

The game already uses metres and seconds; everything is per unit mass, so forces are N/kg
(= m/s^2), energies J/kg and powers W/kg. Scale factors map the game ship onto a real
aircraft: speeds x LV, accelerations x LA, so times x LT = LV / LA and lengths x LL = LV LT.
Anchors: game "1 g" (g_hold, the lift that holds level flight) -> 9.81 m/s^2, and the
game's minimum level speed -> the P-51D's estimated stall speed.
"""
import math
G = 9.81
# game ship (v4 parameters; measured values from v4_bench feel)
g_game, cl_max, cd0, a_max, t_eng, k_x = 27.0, 0.027, 8e-4, 270.0, 8.0, 0.3
psc, store_max, regen = 200.0, 100.0, 10.0
bullet_speed, bullet_life, collision_r, cooldown, reversal = 500.0, 1.0, 10.0, 0.1, 0.21
u_min = math.sqrt(g_game / cl_max)
u_corner = math.sqrt(a_max / cl_max)
L_corner = a_max / (1 + 1) ** 0.25                       # softmin(270, 270)
lat_corner = math.sqrt(L_corner ** 2 - g_game ** 2)      # with level hold
cruise, sus_u, sus_rate = 105.9, 55.6, math.radians(72.0)
# P-51D (approximate; see sources): mass ~4,500 kg, wing 21.8 m^2, 1,490 hp, climb 16.3 m/s,
# top 177 m/s at 5,000 ft (195 at altitude), +8 g limit, corner ~134 m/s (forum), 180 deg in 10 s at 113 m/s (forum),
# stall estimated with C_Lmax ~1.5 (assumption), C_D0 ~0.0163 (commonly cited), M2 .50 muzzle ~870 m/s.
m, S, rho, P_shaft = 4500.0, 21.8, 1.225, 1490 * 745.7
v_stall = math.sqrt(2 * m * G / (rho * S * 1.5))
LV, LA = v_stall / u_min, G / g_game
LT, LL = LV / LA, LV * LV / LA
print(f"scale: speed x{LV:.2f}, acceleration x{LA:.3f}, time x{LT:.2f}, length x{LL:.2f}")
print(f"  (the game runs ~{LT:.1f}x faster than a P-51 and is ~{LL:.1f}x smaller)")
rows = [
    ("minimum level speed (stall)", "m/s", u_min, u_min * LV, f"{v_stall:.0f} (est., C_Lmax 1.5)"),
    ("cruise / level top speed", "m/s", cruise, cruise * LV, "162 cruise; 177 top at 5,000 ft"),
    ("corner speed", "m/s", u_corner, u_corner * LV, "~134 (forum)"),
    ("max lateral accel", "m/s^2 (g)", a_max, a_max * LA, "78.5 (+8 g limit)"),
    ("instant. turn rate at corner", "deg/s", math.degrees(lat_corner / u_corner), math.degrees(lat_corner / u_corner) / LT, f"{math.degrees(78.5 / 134):.0f} (8 g at 134 m/s)"),
    ("sustained turn rate", "deg/s", math.degrees(sus_rate), math.degrees(sus_rate) / LT, "18 (180 deg in 10 s, forum)"),
    ("sustained turn speed", "m/s", sus_u, sus_u * LV, "113"),
    ("turn radius at corner", "m", u_corner ** 2 / lat_corner, u_corner ** 2 / lat_corner * LL, f"{134 ** 2 / math.sqrt(78.5**2 - G**2):.0f}"),
    ("engine accel (thrust/mass)", "m/s^2", t_eng, t_eng * LA, f"{P_shaft * 0.8 / 160 / m:.1f} at 160 m/s (80% prop)"),
    ("drag coefficient per metre", "1/m", cd0, cd0 * LA / LV ** 2, f"{rho * S * 0.0163 / (2 * m):.1e}"),
    ("lift coefficient per metre (max)", "1/m", cl_max, cl_max * LA / LV ** 2, f"{rho * S * 1.5 / (2 * m):.1e}"),
    ("energy store (specific energy)", "J/kg", psc * store_max, psc * store_max * LV ** 2, "-"),
    ("  as altitude band", "m", psc * store_max / g_game, psc * store_max / g_game * LL, "ceiling 12,800"),
    ("regen (specific power)", "W/kg", psc * regen, psc * regen * LV ** 2 / LT, f"{P_shaft / m:.0f} shaft"),
    ("  as climb rate", "m/s", psc * regen / g_game, psc * regen / g_game * LV, "16.3"),
    ("full dive exchange along path", "m/s^2 (g)", k_x * (g_game + L_corner), k_x * (g_game + L_corner) * LA, "<= 9.8 (1 g, vertical dive)"),
    ("full climb exchange along path", "m/s^2 (g)", k_x * (L_corner - g_game), k_x * (L_corner - g_game) * LA, "<= 9.8"),
    ("bullet speed", "m/s", bullet_speed, bullet_speed * LV, "~870 (M2 .50)"),
    ("bullet range (lifetime x speed)", "m", bullet_speed * bullet_life, bullet_speed * bullet_life * LL, "effective ~300-700"),
    ("collision radius", "m", collision_r, collision_r * LL, "half-span 5.6"),
    ("firing cooldown", "s", cooldown, cooldown * LT, "6 guns x ~13/s"),
    ("nose reversal (90%) at corner", "s", reversal, reversal * LT, "no sourced roll figure"),
]
print(f"\n{'quantity':<34} {'unit':<10} {'game':>9} {'game scaled':>12}   P-51D")
for name, unit, game, scaled, ref in rows:
    g_ = f"{game:.3g}" if abs(game) < 1 else f"{game:.1f}"
    s_ = f"{scaled:.3g}" if abs(scaled) < 1 else f"{scaled:.1f}"
    print(f"{name:<34} {unit:<10} {g_:>9} {s_:>12}   {ref}")
