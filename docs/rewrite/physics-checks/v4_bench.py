"""v4 (x/y and PGA, with fields): agreement, accuracy, energy, cost, feel. MODE=acc|cost|feel"""
import os, sys, time, math, dataclasses
sys.path.insert(0, os.path.dirname(__file__))
import jax
MODE = os.environ.get("MODE", "acc")
jax.config.update("jax_enable_x64", MODE != "cost")
import jax.numpy as jnp
import flight4 as F, ship_xy as X, ship_pga as G
SH = F.Ship()
VERS = {"x/y": X, "PGA": G}
S, HOLD, SECONDS, ARENA = 2048, 0.1, 2.0, 2000.0
ks = jax.random.split(jax.random.PRNGKey(0), 12)
n_hold = int(round(SECONDS / HOLD))
acts = jnp.stack([jax.random.randint(ks[0], (n_hold, S), 0, 3), jax.random.randint(ks[1], (n_hold, S), 0, 7)], -1)
speed0 = jax.random.uniform(ks[2], (S,), minval=60.0, maxval=180.0)
head0 = jax.random.uniform(ks[3], (S,), minval=0.0, maxval=2 * math.pi)
power0 = jax.random.uniform(ks[4], (S,), minval=5.0, maxval=100.0)
x0 = jax.random.uniform(ks[5], (S,), minval=-ARENA, maxval=ARENA); y0 = jax.random.uniform(ks[6], (S,), minval=-ARENA, maxval=ARENA)
NF = 24
FIELD = (jax.random.uniform(ks[7], (NF,), minval=-ARENA, maxval=ARENA), jax.random.uniform(ks[8], (NF,), minval=-ARENA, maxval=ARENA),
         jax.random.uniform(ks[9], (NF,), minval=100.0, maxval=490.0), jnp.full((NF,), 40.0),
         math.log(math.sqrt(2.0)) * jax.random.randint(ks[10], (NF,), -2, 3).astype(jnp.float64))
NOFIELD = tuple(f[:0] for f in FIELD)

def cast(t, dt): return jax.tree_util.tree_map(lambda v: jnp.asarray(v, dt), t)

def run(M, field, hz, dt=jnp.float32, sh=SH, a=None):
    a = acts if a is None else a
    fld = cast(field, dt); h = 1.0 / hz; tph = int(round(HOLD * hz))
    st = cast(M.init(*cast((x0, y0, head0, speed0, power0), dt), fld), dt)
    def hold(st, ac):
        return jax.lax.fori_loop(0, tph, lambda i, s: M.step(sh, fld, s, ac, h), st), None
    return jax.jit(lambda st: jax.lax.scan(hold, st, a)[0])(st)

if MODE == "acc":
    for label, field in (("no fields", NOFIELD), ("fields", FIELD)):
        ref = X.pose(run(X, field, 3840, jnp.float64)); refg = G.pose(run(G, field, 3840, jnp.float64))
        print(f"[{label}] x/y vs PGA at 3840 Hz f64: max {float(jnp.max(jnp.hypot(ref[0] - refg[0], ref[1] - refg[1]))):.0e} m")
        for name, M in VERS.items():
            errs = []
            for hz in (30, 60, 120):
                x, y, _ = M.pose(run(M, field, hz)); errs.append(float(jnp.mean(jnp.hypot(x - ref[0], y - ref[1]))))
            print(f"  {name:>4}: mean pos error 30/60/120 Hz = " + " / ".join(f"{e:.3f}" for e in errs) + " m")
        sh0 = dataclasses.replace(SH, cd0=0.0, k_ind=0.0, regen=0.0, t_eng=0.0)
        for name, M in VERS.items():
            fld = cast(field, jnp.float32)
            e0 = M.energy(sh0, cast(M.init(*cast((x0, y0, head0, speed0, power0), jnp.float32), fld), jnp.float32))
            e1 = M.energy(sh0, run(M, field, 30, sh=sh0))
            print(f"  {name:>4}: energy drift (drag, regen, engine off) 30 Hz: {float(jnp.mean(jnp.abs(e1 - e0) / e0)):.1e}")
elif MODE == "cost":
    big = 40960
    k = jax.random.split(jax.random.PRNGKey(1), 6)
    a = jnp.stack([jax.random.randint(k[0], (big,), 0, 3), jax.random.randint(k[1], (big,), 0, 7)], -1)
    xs = jax.random.uniform(k[2], (big,), minval=-2000.0, maxval=2000.0); ys = jax.random.uniform(k[3], (big,), minval=-2000.0, maxval=2000.0)
    hd = jax.random.uniform(k[4], (big,), minval=0.0, maxval=6.28); sp = jax.random.uniform(k[5], (big,), minval=60.0, maxval=180.0)
    F10 = cast(tuple(f[:10] for f in FIELD), jnp.float32)
    for label, Fd in (("0 fields", tuple(f[:0] for f in F10)), ("10 fields", F10)):
        row = []
        for name, M in VERS.items():
            st = cast(M.init(xs, ys, hd, sp, 50.0 + 0 * xs, Fd), jnp.float32)
            f = jax.jit(lambda s: jax.lax.fori_loop(0, 100, lambda i, s: M.step(SH, Fd, s, a, 1 / 30), s))
            jax.block_until_ready(f(st)); ts = []
            for _ in range(9):
                t = time.perf_counter(); jax.block_until_ready(f(st)); ts.append(time.perf_counter() - t)
            row.append(f"{name} {min(ts) / 100 * 1e3:.1f}")
        print(f"{label:>9} (ms per step, 40,960 ships): " + " | ".join(row))
else:  # feel: single ships, x/y, f64, 120 Hz, no fields
    def fly(speed, power, act, seconds, hz=120):
        n = int(round(seconds * hz))
        st = X.init(jnp.array([0.0]), jnp.array([0.0]), jnp.array([0.0]), jnp.array([float(speed)]), jnp.array([float(power)]), NOFIELD)
        a = jnp.array([act])
        out = jax.jit(lambda s: jax.lax.scan(lambda s, i: (X.step(SH, NOFIELD, s, a, 1 / hz), (X.cabs(s[2])[0], s[4][0], s[3][0])), s, None, length=n))(st)
        return out[1]
    print(f"level-flight minimum speed sqrt(g_hold / C_Lmax) = {math.sqrt(SH.g_hold / SH.cl_max):.1f} m/s")
    print("\nsteady state after 6 s holding one input (speed m/s, turn rate deg/s, power change /s):")
    print(f"{'input':>26} {'speed':>6} {'turn':>6} {'power/s':>8}")
    for name, act in (("coast straight", [0, 0]), ("coast full stick", [0, 6]), ("coast 2/3 stick", [0, 4]),
                      ("reverse (pull) straight", [2, 0]), ("reverse + full stick", [2, 6]), ("boost (push) straight", [1, 0]), ("boost + full stick", [1, 6])):
        u, p, r = fly(120.0, 50.0, act, 6.0)
        print(f"{name:>26} {float(u[-1]):6.1f} {math.degrees(float(r[-1])):6.1f} {float(p[-1] - p[-121]):8.2f}")
    print("\nstall: start at 25 m/s, coast straight, power 50 (time to reach 40 m/s, power spent):")
    u, p, r = fly(25.0, 50.0, [0, 0], 6.0)
    i = int(jnp.argmax(u >= 40.0)); print(f"  {i / 120:.2f} s, power {float(p[0] - p[i]):.1f}" if bool(u[i] >= 40.0) else "  never")
    print("stall with an empty store (power 0): speed after 3 s:", round(float(fly(25.0, 0.0, [0, 0], 3.0)[0][-1]), 1), "m/s")
    print("low-speed hard turn: start 45 m/s, coast, full stick, 3 s: speed", round(float(fly(45.0, 50.0, [0, 6], 3.0)[0][-1]), 1), "m/s, power change", round(float(fly(45.0, 50.0, [0, 6], 3.0)[1][-1] - 50.0), 1))
