"""Cost per step (JAX CPU, f32, 40,960 ships), x64 off; plus the simple flight kernel for scale."""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import jax, jax.numpy as jnp
import rigid2d as R
SH, big, W = R.Ship(), 40960, 4096.0
zb = jnp.zeros(big, jnp.float32); ub = 100.0 + zb
a = jnp.stack([jnp.ones(big, jnp.int32), 6 * jnp.ones(big, jnp.int32)], -1)
def bench(step, st, wrap):
    f = jax.jit(lambda s: jax.lax.fori_loop(0, 100, lambda i, s: wrap(step(SH, s, a, 1 / 30)), s))
    jax.block_until_ready(f(st)); ts = []
    for _ in range(7):
        t = time.perf_counter(); jax.block_until_ready(f(st)); ts.append(time.perf_counter() - t)
    return min(ts) / 100 * 1e6
def wrap_xy(s): return (jnp.mod(s[0], W), jnp.mod(s[1], W)) + tuple(s[2:])
def wrap_pga(s):
    x, y = R.pga_apply_origin(s[0]); c, sn = R.pga_heading(s[0])
    return (R.pga_from_pose(jnp.mod(x, W), jnp.mod(y, W), c, sn),) + tuple(s[1:])
print(f"x/y (split): {bench(R.xy_step, (zb, zb, ub, zb, zb, zb, 50.0 + zb), wrap_xy):.0f} us")
print(f"x/y-2 (screw): {bench(R.xy2_step, (zb, zb, 1.0 + zb, zb, zb, ub, zb, 50.0 + zb), wrap_xy):.0f} us")
print(f"PGA (screw): {bench(R.pga_step, (R.pga_from_pose(zb, zb, 1.0 + zb, zb), (zb, ub, zb), 50.0 + zb), wrap_pga):.0f} us")
import today_jax as TJ
c = TJ.Cfg(); a5 = jnp.stack([jnp.ones(big, jnp.int32), 4 * jnp.ones(big, jnp.int32)], -1)
zc = jnp.zeros(big, jnp.complex64)
st = (zc, zc + 100.0, zc + 1.0, zb, zb, 50.0 + zb)
f = jax.jit(lambda s: jax.lax.fori_loop(0, 100, lambda i, s: (lambda o: (jax.lax.complex(jnp.mod(o[0].real, W), jnp.mod(o[0].imag, W)),) + o[1:])(TJ.step(c, s, a5)), s))
jax.block_until_ready(f(st)); ts = []
for _ in range(7):
    t = time.perf_counter(); jax.block_until_ready(f(st)); ts.append(time.perf_counter() - t)
print(f"today's model (JAX port): {min(ts) / 100 * 1e6:.0f} us")
