import os, sys, math, dataclasses
sys.path.insert(0, os.path.dirname(__file__))
os.environ["MODE"] = "none"
src = open(os.path.join(os.path.dirname(__file__), "final_check.py")).read().split('if MODE == "stats":')[0]
exec(src)
u, r, e, al = fly(SH, 105, 0, [0, 0, 0], 12)
print(f"cruise while charging (coast, store filling, 8-12 s): {np.mean(u[8 * HZ:]):.1f} m/s, charge {np.mean(np.diff(e[8 * HZ:])) * HZ:.0f} W")
big = dataclasses.replace(SH, e_max=1e9)
u, r, e, al = fly(big, 105, 0, [2, 0, 0], 40)
print(f"hold reverse, unlimited store: {u[HZ]:.0f} m/s after 1 s, {u[3 * HZ]:.0f} after 3 s; settles {np.mean(u[-HZ:]):.1f} m/s, banking {np.mean(np.diff(e[-HZ:])) * HZ:.0f} W")
