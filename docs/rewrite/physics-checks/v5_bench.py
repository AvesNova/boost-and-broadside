"""v5: feel in SI and P-51 scale, accuracy, energy, cost. MODE=feel|acc|cost"""
import os, sys, time, math, dataclasses
sys.path.insert(0, os.path.dirname(__file__))
import jax
MODE = os.environ.get("MODE", "feel")
jax.config.update("jax_enable_x64", MODE != "cost")
import jax.numpy as jnp
import flight5 as F
SH = F.Ship()
LV, LT = 1.4848, 4.087; LL = LV * LT
NOF = tuple(jnp.zeros((0,)) for _ in range(5))

def fly(speed, E, act, seconds, hz=120, sh=SH):
    n = int(round(seconds * hz))
    st = F.init(jnp.array([0.0]), jnp.array([0.0]), jnp.array([0.0]), jnp.array([float(speed)]), jnp.array([float(E)]), NOF)
    a = jnp.array([act])
    def body(s, i):
        s2 = F.step(sh, NOF, s, a, 1 / hz)
        return s2, (s2[4][0], s2[6][0], s2[7][0])
    return jax.jit(lambda s: jax.lax.scan(body, s, None, length=n))(st)[1]

if MODE == "feel":
    sh = SH
    print(f"stall (min level speed) {math.sqrt(sh.g / sh.cl_max):.1f} m/s (real {math.sqrt(sh.g / sh.cl_max) * LV:.0f}); corner {sh.u_corner:.0f} m/s (real {sh.u_corner * LV:.0f})")
    print(f"engine {sh.p_thrust + sh.p_store:.0f} W = {sh.p_thrust:.0f} W thrust + {sh.p_store:.0f} W store; store {sh.e_max / 1000:.0f} kJ = {sh.e_max / sh.g:.0f} m game altitude ({sh.e_max / sh.g * LL / 1000:.1f} km real)")
    print(f"store charge as climb: {sh.p_store / sh.g:.1f} m/s game ({sh.p_store / sh.g * LV:.1f} m/s real; P-51 16.3)")
    print(f"\n{'steady after 8 s':>26} {'speed m/s (real)':>17} {'turn deg/s (real)':>18} {'store W':>8}")
    for name, act, E0 in (("coast straight", [0, 0], 5000), ("coast full stick", [0, 6], 5000), ("coast 2/3 stick", [0, 4], 5000),
                          ("boost straight (dive)", [1, 0], 9000), ("boost + full stick", [1, 6], 9000),
                          ("reverse straight (climb)", [2, 0], 1000), ("reverse + full stick", [2, 6], 1000)):
        u, r, E = fly(110.0, E0, act, 8.0)
        print(f"{name:>26} {float(u[-1]):7.1f} ({float(u[-1]) * LV:5.0f}) {math.degrees(float(r[-1])):8.1f} ({math.degrees(float(r[-1])) / LT:5.1f}) {float(E[-1] - E[-121]):8.0f}")
    u, r, E = fly(25.0, 5000, [0, 0], 8.0)
    i = int(jnp.argmax(u >= 40.0))
    print(f"\nstall recovery from 25 m/s (coast): 40 m/s after {i / 120:.2f} s ({i / 120 * LT:.1f} s real), store change {float(E[i] - 5000):.0f} J")
    u, r, E = fly(25.0, 0.0, [0, 0], 8.0)
    i = int(jnp.argmax(u >= 40.0))
    print(f"  with an empty store: 40 m/s after {i / 120:.2f} s" if bool(u[i] >= 40.0) else f"  with an empty store: {float(u[-1]):.1f} m/s after 8 s")
    u, r, E = fly(50.0, 5000, [0, 6], 4.0)
    print(f"full-stick turn from 50 m/s, coast, 4 s: speed {float(u[-1]):.1f} m/s, store change {float(E[-1] - 5000):.0f} J")
    u, r, E = fly(60.0, 10000, [1, 0], 30.0)
    print(f"full dive from a full store at 60 m/s: peak {float(u.max()):.0f} m/s (real {float(u.max()) * LV:.0f}; P-51 dive limit ~225)")
    u, r, E = fly(150.0, 0.0, [2, 0], 10.0)
    print(f"zoom climb from 150 m/s, empty store: store {float(E.max()):.0f} J ({float(E.max()) / SH.g:.0f} m game) before speed {float(u[int(jnp.argmax(E))]):.0f} m/s")
elif MODE == "acc":
    S = 2048
    ks = jax.random.split(jax.random.PRNGKey(0), 12)
    acts = jnp.stack([jax.random.randint(ks[0], (20, S), 0, 3), jax.random.randint(ks[1], (20, S), 0, 7)], -1)
    sp = jax.random.uniform(ks[2], (S,), minval=60.0, maxval=150.0); hd = jax.random.uniform(ks[3], (S,), minval=0.0, maxval=6.28)
    E0 = jax.random.uniform(ks[4], (S,), minval=500.0, maxval=9500.0)
    x0 = jax.random.uniform(ks[5], (S,), minval=-2000.0, maxval=2000.0); y0 = jax.random.uniform(ks[6], (S,), minval=-2000.0, maxval=2000.0)
    FLD = (jax.random.uniform(ks[7], (24,), minval=-2000.0, maxval=2000.0), jax.random.uniform(ks[8], (24,), minval=-2000.0, maxval=2000.0),
           jax.random.uniform(ks[9], (24,), minval=100.0, maxval=490.0), jnp.full((24,), 40.0), math.log(math.sqrt(2.0)) * jax.random.randint(ks[10], (24,), -2, 3).astype(jnp.float64))
    def run(field, hz, dt, sh=SH):
        fld = tuple(jnp.asarray(f, dt) for f in field); h = 1.0 / hz; tph = int(round(0.1 * hz))
        st = tuple(jnp.asarray(v, dt) for v in F.init(x0, y0, hd, sp, E0, field))
        def hold(s, a): return jax.lax.fori_loop(0, tph, lambda i, s: F.step(sh, fld, s, a, h), s), None
        return jax.jit(lambda s: jax.lax.scan(hold, s, acts)[0])(st)
    for label, field in (("no fields", tuple(f[:0] for f in FLD)), ("fields", FLD)):
        ref = run(field, 3840, jnp.float64)
        errs = [float(jnp.mean(jnp.hypot(o[0] - ref[0], o[1] - ref[1]))) for o in (run(field, hz, jnp.float32) for hz in (30, 60, 120))]
        sh0 = dataclasses.replace(SH, cd0=0.0, k_ind=0.0, p_thrust=0.0, p_store=0.0)
        st0 = tuple(jnp.asarray(v, jnp.float32) for v in F.init(x0, y0, hd, sp, E0, field))
        e0 = F.energy(sh0, st0); e1 = F.energy(sh0, run(field, 30, jnp.float32, sh0))
        print(f"[{label}] mean pos error 30/60/120 Hz: " + " / ".join(f"{e:.3f}" for e in errs) + f" m; energy drift (no drag, no engine) {float(jnp.mean(jnp.abs(e1 - e0) / e0)):.1e}")
else:
    big = 40960
    k = jax.random.split(jax.random.PRNGKey(1), 8)
    a = jnp.stack([jax.random.randint(k[0], (big,), 0, 3), jax.random.randint(k[1], (big,), 0, 7)], -1)
    xs = jax.random.uniform(k[2], (big,), minval=-2000.0, maxval=2000.0); ys = jax.random.uniform(k[3], (big,), minval=-2000.0, maxval=2000.0)
    F10 = (jax.random.uniform(k[4], (10,), minval=-2000.0, maxval=2000.0), jax.random.uniform(k[5], (10,), minval=-2000.0, maxval=2000.0),
           jax.random.uniform(k[6], (10,), minval=100.0, maxval=490.0), jnp.full((10,), 40.0), jnp.full((10,), 0.3466))
    for label, Fd in (("0 fields", tuple(f[:0] for f in F10)), ("10 fields", F10)):
        st = F.init(xs, ys, 0 * xs + 1.0, 100.0 + 0 * xs, 5000.0 + 0 * xs, Fd)
        f = jax.jit(lambda s: jax.lax.fori_loop(0, 100, lambda i, s: F.step(SH, Fd, s, a, 1 / 30), s))
        jax.block_until_ready(f(st)); ts = []
        for _ in range(9):
            t = time.perf_counter(); jax.block_until_ready(f(st)); ts.append(time.perf_counter() - t)
        print(f"{label}: {min(ts) / 100 * 1e3:.1f} ms per step (40,960 ships, JAX CPU)")
