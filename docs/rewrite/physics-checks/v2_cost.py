"""v2 cost per step (JAX CPU, f32, 40,960 ships): x/y vs PGA, with 0 and 10 fields."""
import os, sys, time, math
sys.path.insert(0, os.path.dirname(__file__))
import jax, jax.numpy as jnp
import rigid2d_v2 as R, today_jax as TJ
SH, big = R.Ship(), 40960
zb = jnp.zeros(big, jnp.float32); ub = 100.0 + zb
a = jnp.stack([jnp.ones(big, jnp.int32), 6 * jnp.ones(big, jnp.int32)], -1)
key = jax.random.split(jax.random.PRNGKey(0), 5)
F10 = (jax.random.uniform(key[0], (10,), minval=-2000.0, maxval=2000.0), jax.random.uniform(key[1], (10,), minval=-2000.0, maxval=2000.0),
       jax.random.uniform(key[2], (10,), minval=100.0, maxval=490.0), jnp.full((10,), 40.0), jnp.full((10,), 0.3466))
F0 = tuple(f[:0] for f in F10)
xs = jax.random.uniform(key[3], (big,), minval=-2000.0, maxval=2000.0); ys = jax.random.uniform(key[4], (big,), minval=-2000.0, maxval=2000.0)

def bench(fn, st, reps=9):
    f = jax.jit(lambda s: jax.lax.fori_loop(0, 100, lambda i, s: fn(s), s))
    jax.block_until_ready(f(st)); ts = []
    for _ in range(reps):
        t = time.perf_counter(); jax.block_until_ready(f(st)); ts.append(time.perf_counter() - t)
    return min(ts) / 100 * 1e6

for label, F in (("0 fields", F0), ("10 fields", F10)):
    n, gx, gy = R.fields_eval(*F, xs, ys)
    xy0 = (xs, ys, 1.0 + zb, zb, zb, ub * n, zb, 50.0 + zb, n, gx, gy)
    pg0 = (R.motor_mul((1.0 + zb, zb, -0.5 * xs, -0.5 * ys), (1.0 + zb, zb, zb, zb)), zb, ub * n, zb, 50.0 + zb, n, gx, gy)
    t_xy = bench(lambda s: R.xy_step(SH, F, s, a, 1 / 30), xy0)
    t_pg = bench(lambda s: R.pga_step(SH, F, s, a, 1 / 30), pg0)
    t_fe = bench(lambda s: (s[0] + 1e-3,) + R.fields_eval(*F, s[0], s[1])[:1] + (s[2],), (xs, ys, zb)) if F[0].shape[0] else 0.0
    print(f"{label:>9}: x/y {t_xy:6.0f} us   PGA {t_pg:6.0f} us   (PGA/x/y {t_pg / t_xy:.2f})   field eval alone {t_fe:.0f} us")
c = TJ.Cfg(); a5 = jnp.stack([jnp.ones(big, jnp.int32), 4 * jnp.ones(big, jnp.int32)], -1)
zc = jnp.zeros(big, jnp.complex64)
print(f"today's model (JAX port, no fields): {bench(lambda s: TJ.step(c, s, a5), (zc, zc + 100.0, zc + 1.0, zb, zb, 50.0 + zb)):.0f} us")
