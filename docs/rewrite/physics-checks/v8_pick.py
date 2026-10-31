import os, sys, math, dataclasses
sys.path.insert(0, os.path.dirname(__file__))
exec(open(os.path.join(os.path.dirname(__file__), "v8_rate.py")).read().split('print("P-51 target')[0])
CANDS = {"v7 (p4, nose x1, 8 g)": (2, 1.0, 216.0), "p16, nose x2, 8 g": (4, 2.0, 216.0),
         "p16, nose x2, 8.6 g": (4, 2.0, 232.0), "p16, nose x2, 9 g": (4, 2.0, 243.0)}
print(f"{'candidate':>22} {'corner':>6} {'peak path deg/s':>15} {'t90 s':>6} {'sustained deg/s @ m/s':>22} {'radius m':>9}")
for name, (k, nm, amax) in CANDS.items():
    sh = dataclasses.replace(make(k, nm), a_max=amax, w_c=12.3 * math.sqrt(amax / 270.0))
    best = (0, 0, 0)
    for u0 in (85, 90, 95, 100, 105, 110):
        u, r, e = fly(sh, u0, 5000.0, [0, 6, 0], 1.0)
        i = int(np.argmax(np.abs(r)))
        if abs(r[i]) > best[0]: best = (abs(r[i]), u[i], int(np.argmax(np.abs(r) >= 0.9 * abs(r[i]))) / HZ)
    u, r, e = fly(sh, 100.0, 5000.0, [0, 6, 0], 15.0)
    us, ws = float(np.mean(u[-HZ:])), float(np.mean(r[-HZ:]))
    print(f"{name:>22} {sh.u_corner:6.0f} {math.degrees(best[0]):15.0f} {best[2]:6.2f} {math.degrees(ws):14.0f} @ {us:5.0f} {us / ws:9.0f}")
# accuracy at 30/60/120 Hz with random inputs, no fields
S = 1024
ks = jax.random.split(jax.random.PRNGKey(0), 8)
acts = jnp.stack([jax.random.randint(ks[0], (20, S), 0, 3), jax.random.randint(ks[1], (20, S), 0, 7), jnp.zeros((20, S), jnp.int32)], -1)
sp0 = jax.random.uniform(ks[2], (S,), minval=60.0, maxval=150.0); hd = jax.random.uniform(ks[3], (S,), minval=0.0, maxval=6.28)
E0 = jax.random.uniform(ks[4], (S,), minval=500.0, maxval=9500.0)
def run(sh, hz, dt):
    h = 1.0 / hz; tph = int(round(0.1 * hz)); z = jnp.zeros(S)
    st = tuple(jnp.asarray(v, dt) for v in F.init(z, z, hd, sp0, E0, NOF)); fl = tuple(jnp.asarray(f, dt) for f in NOF)
    def hold(s, a): return jax.lax.fori_loop(0, tph, lambda i, s: F.step(sh, fl, s, a, h), s), None
    return jax.jit(lambda s: jax.lax.scan(hold, s, acts)[0])(st)
for name, (k, nm, amax) in CANDS.items():
    sh = dataclasses.replace(make(k, nm), a_max=amax, w_c=12.3 * math.sqrt(amax / 270.0))
    ref = run(sh, 3840, jnp.float64)
    errs = [float(jnp.mean(jnp.hypot(o[0] - ref[0], o[1] - ref[1]))) for o in (run(sh, hz, jnp.float32) for hz in (30, 60, 120))]
    print(f"{name:>22}: mean pos error 30/60/120 Hz " + " / ".join(f"{e:.3f}" for e in errs) + " m")
