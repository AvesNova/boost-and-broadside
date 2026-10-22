"""Optimized v5 vs v5 vs today: op count, cost, accuracy vs the v5 fine reference."""
import os, sys, time, math
sys.path.insert(0, os.path.dirname(__file__))
import jax
X64 = os.environ.get("MODE", "cost") == "acc"
jax.config.update("jax_enable_x64", X64)
import jax.numpy as jnp
import flight5 as F, flight5_fast as FF, today_jax as TJ
import opcount_lib as OC
SH = F.Ship()
if not X64:
    z = jnp.zeros(4); NOF = tuple(jnp.zeros((0,)) for _ in range(5)); a = jnp.zeros((4, 2), jnp.int32)
    print("ops per step: v5", OC.count(lambda s: F.step(SH, NOF, s, a, 1 / 30), F.init(z, z, z, z + 100.0, z + 5000.0, NOF))[1],
          "| v5 fast", OC.count(lambda s: FF.step(SH, NOF, s, a, 1 / 30), FF.init(z, z, z, z + 100.0, z + 5000.0, NOF))[1])
    print("  v5 fast detail:", OC.count(lambda s: FF.step(SH, NOF, s, a, 1 / 30), FF.init(z, z, z, z + 100.0, z + 5000.0, NOF))[0])
    big = 40960
    k = jax.random.split(jax.random.PRNGKey(1), 8)
    acts = jnp.stack([jax.random.randint(k[0], (big,), 0, 3), jax.random.randint(k[1], (big,), 0, 7)], -1)
    xs = jax.random.uniform(k[2], (big,), minval=-2000.0, maxval=2000.0); ys = jax.random.uniform(k[3], (big,), minval=-2000.0, maxval=2000.0)
    F10 = (jax.random.uniform(k[4], (10,), minval=-2000.0, maxval=2000.0), jax.random.uniform(k[5], (10,), minval=-2000.0, maxval=2000.0),
           jax.random.uniform(k[6], (10,), minval=100.0, maxval=490.0), jnp.full((10,), 40.0), jnp.full((10,), 0.3466))
    def bench(fn, st):
        f = jax.jit(lambda s: jax.lax.fori_loop(0, 100, lambda i, s: fn(s), s))
        jax.block_until_ready(f(st)); ts = []
        for _ in range(15):
            t = time.perf_counter(); jax.block_until_ready(f(st)); ts.append(time.perf_counter() - t)
        return min(ts) / 100 * 1e3
    c = TJ.Cfg(); a5 = jnp.stack([jnp.ones(big, jnp.int32), 4 * jnp.ones(big, jnp.int32)], -1); zc = jnp.zeros(big, jnp.complex64); zb = jnp.zeros(big)
    print(f"today (no fields): {bench(lambda s: TJ.step(c, s, a5), (zc, zc + 100.0, zc + 1.0, zb, zb, 50.0 + zb)):.1f} ms")
    for label, Fd in (("0 fields", tuple(f[:0] for f in F10)), ("10 fields", F10)):
        t1 = bench(lambda s: F.step(SH, Fd, s, acts, 1 / 30), F.init(xs, ys, 0 * xs + 1.0, 100.0 + 0 * xs, 5000.0 + 0 * xs, Fd))
        t2 = bench(lambda s: FF.step(SH, Fd, s, acts, 1 / 30), FF.init(xs, ys, 0 * xs + 1.0, 100.0 + 0 * xs, 5000.0 + 0 * xs, Fd))
        tf = bench(lambda s: (s[0] + 1e-3, s[1]) + F.fields_eval(*Fd, s[0], s[1])[:1], (xs, ys, zb)) if Fd[0].shape[0] else 0.0
        print(f"{label}: v5 {t1:.1f} ms | v5 fast {t2:.1f} ms | field evaluation alone {tf:.1f} ms")
else:
    S = 2048
    ks = jax.random.split(jax.random.PRNGKey(0), 12)
    acts = jnp.stack([jax.random.randint(ks[0], (20, S), 0, 3), jax.random.randint(ks[1], (20, S), 0, 7)], -1)
    sp = jax.random.uniform(ks[2], (S,), minval=60.0, maxval=150.0); hd = jax.random.uniform(ks[3], (S,), minval=0.0, maxval=6.28)
    E0 = jax.random.uniform(ks[4], (S,), minval=500.0, maxval=9500.0)
    x0 = jax.random.uniform(ks[5], (S,), minval=-2000.0, maxval=2000.0); y0 = jax.random.uniform(ks[6], (S,), minval=-2000.0, maxval=2000.0)
    FLD = (jax.random.uniform(ks[7], (24,), minval=-2000.0, maxval=2000.0), jax.random.uniform(ks[8], (24,), minval=-2000.0, maxval=2000.0),
           jax.random.uniform(ks[9], (24,), minval=100.0, maxval=490.0), jnp.full((24,), 40.0), math.log(math.sqrt(2.0)) * jax.random.randint(ks[10], (24,), -2, 3).astype(jnp.float64))
    def run(M, field, hz, dt):
        fld = tuple(jnp.asarray(f, dt) for f in field); h = 1.0 / hz; tph = int(round(0.1 * hz))
        st = tuple(jnp.asarray(v, dt) for v in M.init(x0, y0, hd, sp, E0, field))
        def hold(s, a): return jax.lax.fori_loop(0, tph, lambda i, s: M.step(SH, fld, s, a, h), s), None
        return jax.jit(lambda s: jax.lax.scan(hold, s, acts)[0])(st)
    for label, field in (("no fields", tuple(f[:0] for f in FLD)), ("fields", FLD)):
        ref = run(F, field, 3840, jnp.float64)
        for name, M in (("v5", F), ("v5 fast", FF)):
            errs = [float(jnp.mean(jnp.hypot(o[0] - ref[0], o[1] - ref[1]))) for o in (run(M, field, hz, jnp.float32) for hz in (30, 60, 120))]
            print(f"[{label}] {name:>8}: mean pos error 30/60/120 Hz " + " / ".join(f"{e:.3f}" for e in errs) + " m")
