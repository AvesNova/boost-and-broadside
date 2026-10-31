"""Wrap cost: x/y and PGA screw steps, without wrap, with a naive PGA wrap, and with a
translator-composition PGA wrap (M' = T(-k W) M). JAX CPU, f32, 40,960 ships."""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import jax, jax.numpy as jnp
import rigid2d as R, today_jax as TJ
SH, big, W = R.Ship(), 40960, 4096.0
zb = jnp.zeros(big, jnp.float32); ub = 100.0 + zb
a = jnp.stack([jnp.ones(big, jnp.int32), 6 * jnp.ones(big, jnp.int32)], -1)

def bench(stepfn, st):
    f = jax.jit(lambda s: jax.lax.fori_loop(0, 100, lambda i, s: stepfn(s), s))
    jax.block_until_ready(f(st)); ts = []
    for _ in range(9):
        t = time.perf_counter(); jax.block_until_ready(f(st)); ts.append(time.perf_counter() - t)
    return min(ts) / 100 * 1e6

def wrap_xy(s): return (jnp.mod(s[0], W), jnp.mod(s[1], W)) + tuple(s[2:])

def wrap_pga_naive(s):  # extract pose with atan2, wrap, rebuild with half-angle cos/sin
    x, y = R.pga_apply_origin(s[0]); c, sn = R.pga_heading(s[0])
    return (R.pga_from_pose(jnp.mod(x, W), jnp.mod(y, W), c, sn),) + tuple(s[1:])

def wrap_pga_translator(s):  # M' = T(-k W) M: only the translation components change
    m, x_y = s[0], R.pga_apply_origin(s[0])
    kx, ky = jnp.floor(x_y[0] / W) * W, jnp.floor(x_y[1] / W) * W
    s_, p, a_, b = m
    ta, tb = 0.5 * kx, 0.5 * ky
    return ((s_, p, a_ + ta * s_ - tb * p, b + ta * p + tb * s_),) + tuple(s[1:])

xy0 = (zb, zb, 1.0 + zb, zb, zb, ub, zb, 50.0 + zb)
pg0 = (R.pga_from_pose(zb, zb, 1.0 + zb, zb), (zb, ub, zb), 50.0 + zb)
step_xy = lambda s: R.xy2_step(SH, s, a, 1 / 30)
step_pg = lambda s: R.pga_step(SH, s, a, 1 / 30)
c = TJ.Cfg(); a5 = jnp.stack([jnp.ones(big, jnp.int32), 4 * jnp.ones(big, jnp.int32)], -1)
zc = jnp.zeros(big, jnp.complex64)
tj0 = (zc, zc + 100.0, zc + 1.0, zb, zb, 50.0 + zb)
step_tj = lambda s: TJ.step(c, s, a5)
wrap_tj = lambda o: (jax.lax.complex(jnp.mod(o[0].real, W), jnp.mod(o[0].imag, W)),) + o[1:]
rows = [("today (JAX port)", bench(step_tj, tj0), bench(lambda s: wrap_tj(step_tj(s)), tj0), None),
        ("rigid body, x/y", bench(step_xy, xy0), bench(lambda s: wrap_xy(step_xy(s)), xy0), None),
        ("rigid body, PGA", bench(step_pg, pg0), bench(lambda s: wrap_pga_naive(step_pg(s)), pg0),
         bench(lambda s: wrap_pga_translator(step_pg(s)), pg0))]
print(f"{'model':>18} {'no wrap us':>10} {'wrap us':>8} {'translator wrap us':>18}")
for name, nw, w, tw in rows:
    print(f"{name:>18} {nw:10.0f} {w:8.0f} {'' if tw is None else f'{tw:.0f}':>18}")
# correctness of the translator wrap
m = R.pga_from_pose(jnp.array([5000.0, -10.0]), jnp.array([-300.0, 9000.0]), jnp.array([0.6, 0.0]), jnp.array([0.8, 1.0]))
w = wrap_pga_translator((m, None, None))[0]
print("translator wrap check:", [float(v) for v in R.pga_apply_origin(w)[0]], [float(v) for v in R.pga_apply_origin(w)[1]], "heading kept:", bool(jnp.allclose(jnp.stack(R.pga_heading(w)), jnp.stack(R.pga_heading(m)))))
