import os, sys, math, dataclasses
sys.path.insert(0, os.path.dirname(__file__))
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy as np
import flight9 as F
HZ = 30
NOF = tuple(jnp.zeros((0,)) for _ in range(5))
def ship(**over):
    b = F.Ship()
    k_ind, power, fs, a0, u_c = 3.0 * b.k_ind, 1200.0, 20.0, 3.0, 105.0
    cd0 = (min(power / u_c, fs) - k_ind * (b.g / u_c ** 2) ** 2 * u_c ** 2 - a0) / u_c ** 2
    sh = dataclasses.replace(b, k_ind=k_ind, power=power, f_static=fs, a0=a0, cd0=cd0, soft_k=4, nose_mult=2.0, bullet_cost=29.1)
    return dataclasses.replace(sh, **over)
def fly(sh, speed, E, act, sec):
    st = F.init(jnp.array([0.0]), jnp.array([0.0]), jnp.array([0.0]), jnp.array([float(speed)]), jnp.array([float(E)]), NOF)
    a = jnp.array([act])
    def body(s, i):
        s2 = F.step(sh, NOF, s, a, 1 / HZ)
        return s2, (s2[4][0], jnp.arctan2(s2[3][0], s2[2][0]) - s2[5][0], s2[9][0], s2[5][0])
    u, th, e, al = jax.jit(lambda s: jax.lax.scan(body, s, None, length=int(sec * HZ)))(st)[1]
    th = np.unwrap(np.asarray(th))
    return np.asarray(u), np.diff(th, prepend=th[0]) * HZ, np.asarray(e), np.asarray(al)
def tail(x, sec=2): return float(np.mean(x[-int(sec * HZ):]))
