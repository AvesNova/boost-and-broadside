"""Per-coefficient behaviour for lift-coefficient turn levels (final config)."""
from v9_common import *
rows = []
for c in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
    F.STICK = jnp.array([0.0, -c, c, -c, c, -c, c])   # every turn action = this coefficient
    sh = ship(levels="cl"); big = dataclasses.replace(sh, e_max=1e9)
    ua, ra, ea, aa = fly(sh, 50.0, sh.e_max, [0, 6, 0], 40)
    ub, rb, eb, ab = fly(sh, 130.0, sh.e_max, [0, 6, 0], 40)
    uc, rc, ec, ac = fly(big, 90.0, 5e8, [1, 6, 0], 40)
    u1, r1, e1, a1 = fly(sh, 120.0, 5000.0, [0, 6, 0], 2.0)
    stable = abs(tail(ua) - tail(ub)) < 0.5
    rows.append((c, tail(ua), math.degrees(tail(ra)), tail(ua) / max(tail(ra), 1e-9), math.degrees(tail(aa)), stable,
                 math.degrees(tail(rc)), tail(uc), math.degrees(np.abs(r1).max()), u1[-1]))
for c, u, rate, rad, off, st, brate, bu, pk, u2 in rows:
    print(f"C_L {c:.1f}: coast {u:.0f} m/s, {rate:.0f} deg/s, r {rad:.0f} m, gun {off:.1f} deg{'' if st else ' (NOT stable)'} | "
          f"boost {brate:.0f} deg/s at {bu:.0f} | from 120: {pk:.0f} deg/s, {u2:.0f} m/s at 2 s | min radius 1/C_L: {37 / c:.0f} m")
