from v9_common import *
for levels in ("lift", "stick"):
    sh = ship(levels=levels)
    big = dataclasses.replace(sh, e_max=1e9)
    print(f"\n=== levels = {levels}")
    print(f"{'level':>5} | {'coast datum (from 50 / 120 m/s)':>36} {'deg/s':>6} {'r m':>5} {'offset':>6} | {'boost datum, unlimited store':>28} {'deg/s':>6}")
    for lvl, st in ((1 / 3, 2), (2 / 3, 4), (1.0, 6)):
        ua, ra, ea, aa = fly(sh, 50.0, sh.e_max, [0, st, 0], 40)
        ub, rb, eb, ab = fly(sh, 120.0, sh.e_max, [0, st, 0], 40)
        uc, rc, ec, ac = fly(big, 90.0, 5e8, [1, st, 0], 40)
        print(f"{lvl:5.2f} | {tail(ua):17.1f} / {tail(ub):5.1f} m/s           {math.degrees(tail(ra)):6.1f} {tail(ua) / tail(ra):5.0f} {math.degrees(tail(aa)):5.1f}d | "
              f"{tail(uc):21.1f} m/s     {math.degrees(tail(rc)):6.1f}")
    print("  instantaneous peak path rate (deg/s) starting at each speed:")
    for lvl, st in ((1 / 3, 2), (2 / 3, 4), (1.0, 6)):
        row = []
        for u0 in (40, 60, 90, 120, 140):
            u, r, e, al = fly(sh, float(u0), 5000.0, [0, st, 0], 0.8)
            row.append(f"{u0}: {math.degrees(np.abs(r).max()):4.0f}")
        print(f"   {lvl:4.2f}  " + "  ".join(row))
