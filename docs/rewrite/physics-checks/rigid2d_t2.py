import os, sys, math
sys.path.insert(0, os.path.dirname(__file__))
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import rigid2d as R
sh = R.Ship()
for head in (0.0, 0.7, 2.5):
  for act in ([1, 6], [0, 3]):
    a = jnp.array(act); c, s = math.cos(head), math.sin(head)
    xy = tuple(jnp.array(v) for v in (0.0, 0.0, 120.0 * c, 120.0 * s, head, 0.0, 50.0))
    pg = (R.pga_from_pose(jnp.array(0.0), jnp.array(0.0), jnp.array(c), jnp.array(s)), (jnp.array(0.0), jnp.array(120.0), jnp.array(0.0)), jnp.array(50.0))
    f = jax.jit(lambda xy, pg: jax.lax.fori_loop(0, 2000, lambda i, t: (R.xy_step(sh, t[0], a, 1/4000), R.pga_step(sh, t[1], a, 1/4000)), (xy, pg)))
    xy, pg = f(xy, pg)
    print(head, act, "xy", [round(float(v), 2) for v in (xy[0], xy[1], xy[4])], "pga", [round(float(v), 2) for v in (*R.pga_apply_origin(pg[0]), math.atan2(*R.pga_heading(pg[0])[::-1]))])
