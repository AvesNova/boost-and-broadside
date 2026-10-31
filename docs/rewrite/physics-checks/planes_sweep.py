"""Per-plane, per-coefficient turn behaviour (lift-coefficient levels)."""
import json
from v9_common import *
import planes as PL
CS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0)
res = {}
for name in ("ours", "P-51D", "Spitfire IX"):
    sh = PL.plane(name); big = dataclasses.replace(sh, e_max=1e9)
    rows = []
    for c in CS:
        F.STICK = jnp.array([0.0, -c, c, -c, c, -c, c])
        ua, ra, ea, aa = fly(sh, 60.0, sh.e_max, [0, 6, 0], 40)        # coast, store full: level turn
        ub, rb, eb, ab = fly(sh, 120.0, sh.e_max, [0, 6, 0], 40)
        uc, rc, ec, ac = fly(big, 90.0, 5e8, [1, 6, 0], 40)              # boost, unlimited store
        rows.append(dict(c=c, cu=tail(ua), cr=math.degrees(tail(ra)), crad=tail(ua) / max(tail(ra), 1e-9),
                         stable=abs(tail(ua) - tail(ub)) < 1.0, bu=tail(uc), br=math.degrees(tail(rc)),
                         brad=tail(uc) / max(tail(rc), 1e-9)))
    res[name] = rows
json.dump(res, open(os.path.join(os.path.dirname(__file__), "planes_sweep.json"), "w"), indent=1)
for name, rows in res.items():
    print(f"\n## {name}")
    for r in rows:
        print(f"- C_L {r['c']:.2f}: coast {r['cu']:.0f} m/s, {r['cr']:.0f} deg/s, r {r['crad']:.0f} m{'' if r['stable'] else ' (unstable)'}; "
              f"boost {r['br']:.0f} deg/s at {r['bu']:.0f} m/s, r {r['brad']:.0f} m")
