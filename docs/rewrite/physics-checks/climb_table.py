import os, sys, math
sys.path.insert(0, os.path.dirname(__file__))
import jax.numpy as jnp
import rigid2d_v3 as R
sh = R.Ship()
print(f"{'u m/s':>6} {'lift budget m/s2':>16} {'climb, straight':>15} {'climb, 1/3 stick':>16} {'climb, 2/3 stick':>16} {'climb, full stick':>17}")
for u in (10, 20, 30, 40, 60, 80, 100, 140, 180):
    u = jnp.float32(u)
    row = []
    for stick in (0.0, 1 / 3, 2 / 3, 1.0):
        a = R.alpha_cmd(sh, jnp.float32(stick), u)
        a_lat = jnp.abs(R.lift(sh, a)) * u * u
        row.append(float(-R.thrust_eff(sh, jnp.float32(-80.0), jnp.float32(50.0), u, a_lat)))
    print(f"{float(u):>6.0f} {float(R.budget(sh, u)):16.1f} " + " ".join(f"{v:>16.1f}" for v in row))
