from v9_common import *
OPTIONS = (("lift", "Lift thirds"), ("angle", "Angle thirds"), ("cl", "Coefficient thirds"), ("neutral", "Energy-neutral middle"))
out = {}
for key, title in OPTIONS:
    sh = ship(levels=key); big = dataclasses.replace(sh, e_max=1e9)
    rows = []
    for lvl, st in (("1/3", 2), ("2/3", 4), ("3/3", 6)):
        ua, ra, ea, aa = fly(sh, 50.0, sh.e_max, [0, st, 0], 40)
        ub, rb, eb, ab = fly(sh, 120.0, sh.e_max, [0, st, 0], 40)
        uc, rc, ec, ac = fly(big, 90.0, 5e8, [1, st, 0], 40)
        u1, r1, e1, a1 = fly(sh, 120.0, 5000.0, [0, st, 0], 0.8)
        u2, r2, e2, a2 = fly(sh, 120.0, 5000.0, [0, st, 0], 2.0)
        rows.append((lvl, tail(ua), tail(ub), math.degrees(tail(ra)), tail(ua) / max(abs(tail(ra)), 1e-9), math.degrees(tail(aa)),
                     tail(uc), math.degrees(tail(rc)), math.degrees(np.abs(r1).max()), u2[-1]))
    out[title] = rows
for title, rows in out.items():
    print(f"\n## {title}")
    for lvl, ua, ub, rate, rad, off, ubst, rbst, inst120, u2 in rows:
        same = "stable" if abs(ua - ub) < 0.5 else f"from 50: {ua:.0f}, from 120: {ub:.0f}"
        print(f"- {lvl}: coast settles {ua:.0f} m/s ({same}), {rate:.0f} deg/s, radius {rad:.0f} m, gun {off:.1f} deg; "
              f"boost {rbst:.0f} deg/s at {ubst:.0f} m/s; from 120 m/s: peak {inst120:.0f} deg/s, {u2:.0f} m/s after 2 s")
