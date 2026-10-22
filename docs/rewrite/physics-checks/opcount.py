"""Count expensive primitives in one compiled step (jaxpr), for v5 and today's model."""
import os, sys, collections
sys.path.insert(0, os.path.dirname(__file__))
import jax, jax.numpy as jnp
import flight5 as F, today_jax as TJ
EXP = {"sin", "cos", "tan", "atan2", "exp", "log", "log1p", "sqrt", "rsqrt", "pow", "tanh", "sinh", "cosh", "div", "asin", "acos", "integer_pow", "expm1"}
def count(fn, *args):
    c = collections.Counter()
    def walk(j):
        for e in j.eqns:
            if e.primitive.name in EXP: c[e.primitive.name] += 1
            for k, v in e.params.items():
                if hasattr(v, "eqns"): walk(v)
                elif hasattr(v, "jaxpr") and hasattr(v.jaxpr, "eqns"): walk(v.jaxpr)
    walk(jax.make_jaxpr(fn)(*args).jaxpr)
    return dict(sorted(c.items())), sum(c.values())
z = jnp.zeros(4)
NOF = tuple(jnp.zeros((0,)) for _ in range(5))
st = F.init(z, z, z, z + 100.0, z + 5000.0, NOF)
a = jnp.zeros((4, 2), jnp.int32)
print("v5 step:", count(lambda s: F.step(F.Ship(), NOF, s, a, 1 / 30), st))
c = TJ.Cfg(); zc = jnp.zeros(4, jnp.complex64)
print("today step:", count(lambda s: TJ.step(c, s, a + jnp.array([0, 0])), (zc, zc + 100.0, zc + 1.0, z, z, z + 50.0)))
