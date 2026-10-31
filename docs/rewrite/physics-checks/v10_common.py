"""Engine power falling with the store (altitude analog): kinked vs smooth curves (setup).
p_min = the power that just holds cruise at a full store (cruise thrust); knee at half the store."""
from v9_common import *
F.STICK = jnp.array([0.0, -0.4, 0.4, -0.75, 0.75, -1.0, 1.0])
base = ship(levels="cl")
U_CR = 103.72                                         # charging coast cruise of the base ship
T = base.cd0 * U_CR ** 2 + base.k_ind * base.g ** 2 / U_CR ** 2        # level drag at cruise
P_MIN = U_CR * T / (1 - (T / base.f_static) ** 4) ** 0.25           # invert the soft static-thrust cap
CURVES = {"none": base, **{c: dataclasses.replace(base, pcurve=c, p_min=P_MIN, e_knee=0.5)
                          for c in ("kink", "smooth", "smooth2")}}
print(f"p_min {P_MIN:.0f} W (engine {base.power:.0f} W)")

def run(sh, acts, u0, E0, hz):
    st = F.init(*(jnp.array([v]) for v in (0.0, 0.0, 0.0, u0, E0)), NOF)
    seq = jnp.repeat(jnp.array(acts), hz // 30 * 30 * 2, axis=0)[:, None, :]   # each action held 2 s
    def body(s, a):
        s2 = F.step(sh, NOF, s, a, 1.0 / hz)
        return s2, (s2[4][0], s2[9][0])
    s, (u, E) = jax.jit(lambda s, q: jax.lax.scan(body, s, q))(st, seq)
    return np.asarray(u), np.asarray(E), s

