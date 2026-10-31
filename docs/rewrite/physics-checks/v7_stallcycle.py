"""Reverse without the adaptive blend: does it run to stall, and what does the store gain on average?"""
import os, sys, math, dataclasses
sys.path.insert(0, os.path.dirname(__file__))
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import flight7 as F7, flight7_noblend as FN
NOF = tuple(jnp.zeros((0,)) for _ in range(5)); HZ = 30
def make(M, power=1200.0, fs=20.0, a0=3.0, u_c=105.0):
    b = M.Ship(); ind = b.k_ind * (b.g / u_c ** 2) ** 2 * u_c ** 2
    return dataclasses.replace(b, power=power, f_static=fs, a0=a0, cd0=(min(power / u_c, fs) - ind - a0) / u_c ** 2, e_max=1e9)
def fly(M, sh, acts, speed=105.0):
    st = M.init(jnp.array([0.0]), jnp.array([0.0]), jnp.array([0.0]), jnp.array([speed]), jnp.array([1e6]), NOF)
    def body(s, a):
        s2 = M.step(sh, NOF, s, a[None], 1 / HZ)
        return s2, (s2[4][0], s2[9][0])
    return jax.jit(lambda s: jax.lax.scan(body, s, acts))(st)[1]
n = 60 * HZ
rev = jnp.tile(jnp.array([[2, 0, 0]]), (n, 1))
alt = jnp.array([[2, 0, 0] if (i // 15) % 2 else [0, 0, 0] for i in range(n)])  # 0.5 s reverse / 0.5 s coast
for label, M in (("v7 (blend)", F7), ("no blend", FN)):
    sh = make(M)
    for name, acts in (("hold reverse", rev), ("alternate 0.5 s reverse / 0.5 s coast", alt)):
        u, E = fly(M, sh, acts)
        u, E = np.asarray(u), np.asarray(E)
        last = slice(n // 2, n)
        print(f"{label:>11}, {name:<38}: speed after 2 s {u[2 * HZ]:.0f}, min {u.min():.0f} m/s, last 30 s range {u[last].min():.0f}-{u[last].max():.0f} m/s, store gain {(E[-1] - E[n // 2]) / 30:.0f} W")
print("max excess power of this engine: ~804 W at ~62 m/s")
