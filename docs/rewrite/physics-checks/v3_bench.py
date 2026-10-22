"""v3: accuracy (one variant vs reference, plus cross-variant agreement), cost per variant."""
import os, sys, time, math
sys.path.insert(0, os.path.dirname(__file__))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import rigid2d_v3 as R

SH = R.Ship()
S, HOLD, SECONDS, ARENA = 2048, 0.1, 2.0, 2000.0
ks = jax.random.split(jax.random.PRNGKey(0), 12)
n_hold = int(round(SECONDS / HOLD))
acts = jnp.stack([jax.random.randint(ks[0], (n_hold, S), 0, 3), jax.random.randint(ks[1], (n_hold, S), 0, 7)], -1)
speed0 = jax.random.uniform(ks[2], (S,), minval=60.0, maxval=180.0)
head0 = jax.random.uniform(ks[3], (S,), minval=0.0, maxval=2 * math.pi)
power0 = jax.random.uniform(ks[4], (S,), minval=5.0, maxval=100.0)
x0 = jax.random.uniform(ks[5], (S,), minval=-ARENA, maxval=ARENA)
y0 = jax.random.uniform(ks[6], (S,), minval=-ARENA, maxval=ARENA)
NF = 24
FIELD = (jax.random.uniform(ks[7], (NF,), minval=-ARENA, maxval=ARENA), jax.random.uniform(ks[8], (NF,), minval=-ARENA, maxval=ARENA),
         jax.random.uniform(ks[9], (NF,), minval=100.0, maxval=490.0), jnp.full((NF,), 40.0),
         math.log(math.sqrt(2.0)) * jax.random.randint(ks[10], (NF,), -2, 3).astype(jnp.float64))
NOFIELD = tuple(f[:0] for f in FIELD)

def run(variant, field, hz, dt=jnp.float32, sh=SH):
    step = R.STEPS[variant]; fld = tuple(f.astype(dt) for f in field)
    h = 1.0 / hz; tph = int(round(HOLD * hz))
    def hold(st, a):
        return jax.lax.fori_loop(0, tph, lambda i, s: step(sh, fld, s, a, h), st), None
    return jax.jit(lambda st: jax.lax.scan(hold, st, acts)[0])(R.init(variant, x0, y0, head0, speed0, power0, field, dt))

mode = os.environ.get("MODE", "accuracy")
if mode == "accuracy":
    for label, field in (("no fields", NOFIELD), ("fields", FIELD)):
        ref = R.pose("cs_polar", run("cs_polar", field, 3840, jnp.float64))
        print(f"[{label}] reference: cs_polar at 3840 Hz, f64")
        for v in R.STEPS:
            p = R.pose(v, run(v, field, 3840, jnp.float64)) if v != "cs_polar" else ref
            agree = float(jnp.max(jnp.hypot(p[0] - ref[0], p[1] - ref[1])))
            errs = []
            for hz in (30, 60, 120):
                x, y, _ = R.pose(v, run(v, field, hz))
                errs.append(float(jnp.mean(jnp.hypot(x - ref[0], y - ref[1]))))
            print(f"  {v:>12}: agrees with ref to {agree:.0e} m; mean error 30/60/120 Hz: " + " / ".join(f"{e:.3f}" for e in errs) + " m")
else:
    big = 40960
    k = jax.random.split(jax.random.PRNGKey(1), 6)
    a = jnp.stack([jax.random.randint(k[0], (big,), 0, 3), jax.random.randint(k[1], (big,), 0, 7)], -1)
    xs = jax.random.uniform(k[2], (big,), minval=-2000.0, maxval=2000.0); ys = jax.random.uniform(k[3], (big,), minval=-2000.0, maxval=2000.0)
    hd = jax.random.uniform(k[4], (big,), minval=0.0, maxval=6.28); sp = jax.random.uniform(k[5], (big,), minval=60.0, maxval=180.0)
    F10 = tuple(f[:10].astype(jnp.float32) for f in FIELD)
    for label, F in (("0 fields", tuple(f[:0] for f in F10)), ("10 fields", F10)):
        row = []
        for v, step in R.STEPS.items():
            st = R.init(v, xs, ys, hd, sp, 50.0 + 0 * xs, F, jnp.float32)
            f = jax.jit(lambda s: jax.lax.fori_loop(0, 100, lambda i, s: step(SH, F, s, a, 1 / 30), s))
            jax.block_until_ready(f(st)); ts = []
            for _ in range(9):
                t = time.perf_counter(); jax.block_until_ready(f(st)); ts.append(time.perf_counter() - t)
            row.append(f"{v} {min(ts) / 100 * 1e3:.1f}")
        print(f"{label:>9} (ms/step, 40,960 ships): " + " | ".join(row))
