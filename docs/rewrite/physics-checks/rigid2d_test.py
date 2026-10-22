import os, sys, math
sys.path.insert(0, os.path.dirname(__file__))
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import rigid2d as R
# 1) motor exp: constant body twist r=1, v=(1,0) for t=1 -> arc (sin 1, 1-cos 1), heading 1
M = R.pga_from_pose(jnp.array(0.0), jnp.array(0.0), jnp.array(1.0), jnp.array(0.0))
E = R.pga_exp_twist(jnp.array(1.0), jnp.array(1.0), jnp.array(0.0), 1.0)
M1 = R.pga_mul(M, E)
print("exp arc:", R.pga_apply_origin(M1), R.pga_heading(M1), "expect", (math.sin(1), 1 - math.cos(1)), (math.cos(1), math.sin(1)))
# 2) pose round trip
M = R.pga_from_pose(jnp.array(3.0), jnp.array(-2.0), jnp.array(math.cos(0.7)), jnp.array(math.sin(0.7)))
print("pose:", R.pga_apply_origin(M), R.pga_heading(M), "expect (3,-2)", (math.cos(.7), math.sin(.7)))
# 3) single ship, both versions, same action, fine step: compare after 0.5 s
sh = R.Ship()
for act in ([0, 0], [1, 6], [2, 5], [0, 2]):
    a = jnp.array(act)
    xy = (0.0, 0.0, 120.0, 0.0, 0.0, 0.0, 50.0)
    xy = tuple(jnp.array(v) for v in xy)
    pg = (R.pga_from_pose(jnp.array(0.0), jnp.array(0.0), jnp.array(1.0), jnp.array(0.0)), (jnp.array(0.0), jnp.array(120.0), jnp.array(0.0)), jnp.array(50.0))
    h = 1 / 4000
    for _ in range(2000):
        xy = R.xy_step(sh, xy, a, h); pg = R.pga_step(sh, pg, a, h)
    print(act, "xy", [round(float(v), 3) for v in (xy[0], xy[1], xy[4], xy[5])],
          "pga", [round(float(v), 3) for v in (*R.pga_apply_origin(pg[0]), math.atan2(*R.pga_heading(pg[0])[::-1]), pg[1][0])])
