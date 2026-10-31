"""Instantaneous turn rate vs soft g-limit sharpness and nose speed (v8, 1200 W config)."""
import os, sys, math, dataclasses
sys.path.insert(0, os.path.dirname(__file__))
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy as np
import flight8 as F
NOF = tuple(jnp.zeros((0,)) for _ in range(5)); HZ = 30
def make(soft_k, nose_mult, power=1200.0, fs=20.0, a0=3.0, u_c=105.0):
    b = F.Ship(); ind = b.k_ind * (b.g / u_c ** 2) ** 2 * u_c ** 2
    return dataclasses.replace(b, power=power, f_static=fs, a0=a0, cd0=(min(power / u_c, fs) - ind - a0) / u_c ** 2, soft_k=soft_k, nose_mult=nose_mult)
def fly(sh, speed, E, act, sec, hz=HZ):
    st = F.init(jnp.array([0.0]), jnp.array([0.0]), jnp.array([0.0]), jnp.array([float(speed)]), jnp.array([float(E)]), NOF)
    a = jnp.array([act])
    def body(s, i):
        s2 = F.step(sh, NOF, s, a, 1 / hz)
        theta = jnp.arctan2(s2[3][0], s2[2][0]) - s2[5][0]          # flight path = heading - alpha
        return s2, (s2[4][0], theta, s2[9][0])
    u, th, e = jax.jit(lambda s: jax.lax.scan(body, s, None, length=int(sec * hz)))(st)[1]
    th = np.unwrap(np.asarray(th))
    r = np.diff(th, prepend=th[0]) * hz                              # path turn rate
    return np.asarray(u), r, np.asarray(e)
print("P-51 target at game scale: ~137 deg/s instantaneous (8 g at corner)")
print(f"{'soft p':>6} {'nose x':>6} {'cap deg/s':>9} {'peak deg/s':>10} {'at m/s':>7} {'t90 s':>6} {'sustained coast deg/s':>21}")
for k in (2, 3, 4):
    for nm in (1.0, 1.5, 2.0, 3.0):
        sh = make(k, nm)
        L = sh.a_max * 2 ** (-1 / 2 ** k)
        cap = math.degrees(math.sqrt(L * L - sh.g * sh.g) / sh.u_corner)
        best = (0, 0, 0)
        for u0 in (85, 95, 105, 115):
            u, r, e = fly(sh, u0, 5000.0, [0, 6, 0], 1.0)
            i = int(np.argmax(np.abs(r)))
            if abs(r[i]) > best[0]:
                best = (abs(r[i]), u[i], int(np.argmax(np.abs(r) >= 0.9 * abs(r[i]))) / HZ)
        u, r, e = fly(sh, 100.0, 5000.0, [0, 6, 0], 15.0)
        print(f"{2 ** k:>6} {nm:>6.1f} {cap:9.0f} {math.degrees(best[0]):10.0f} {best[1]:7.0f} {best[2]:6.2f} {math.degrees(np.mean(r[-HZ:])):21.0f}")
