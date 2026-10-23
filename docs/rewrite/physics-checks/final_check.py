"""End-to-end check of the proposed flight model (v8 + induced drag x3).
MODE=stats|acc|cost"""
import os, sys, math, dataclasses, time
sys.path.insert(0, os.path.dirname(__file__))
import jax
MODE = os.environ.get("MODE", "stats")
jax.config.update("jax_enable_x64", MODE != "cost")
import jax.numpy as jnp, numpy as np
import flight8 as F
HZ = 30
NOF = tuple(jnp.zeros((0,)) for _ in range(5))

def final_ship():
    b = F.Ship()
    k_ind, power, fs, a0, u_c = 3.0 * b.k_ind, 1200.0, 20.0, 3.0, 105.0
    cd0 = (min(power / u_c, fs) - k_ind * (b.g / u_c ** 2) ** 2 * u_c ** 2 - a0) / u_c ** 2
    sh = dataclasses.replace(b, k_ind=k_ind, power=power, f_static=fs, a0=a0, cd0=cd0, soft_k=4, nose_mult=2.0)
    us = np.linspace(32.0, 140.0, 2000)
    x = (sh.power / us) / sh.f_static; T = (sh.power / us) / (1 + x ** 4) ** 0.25
    pe = float(((T - (sh.cd0 + sh.k_ind * (sh.g / us ** 2) ** 2) * us ** 2) * us).max())
    return dataclasses.replace(sh, bullet_cost=1.2 * pe / HZ), pe

SH, PE = final_ship()

def fly(sh, speed, E, act, sec, field=NOF, hz=HZ):
    st = F.init(jnp.array([0.0]), jnp.array([0.0]), jnp.array([0.0]), jnp.array([float(speed)]), jnp.array([float(E)]), field)
    a = jnp.array([act])
    def body(s, i):
        s2 = F.step(sh, field, s, a, 1 / hz)
        return s2, (s2[4][0], jnp.arctan2(s2[3][0], s2[2][0]) - s2[5][0], s2[9][0], s2[5][0])
    u, th, e, al = jax.jit(lambda s: jax.lax.scan(body, s, None, length=int(sec * hz)))(st)[1]
    th = np.unwrap(np.asarray(th))
    return np.asarray(u), np.diff(th, prepend=th[0]) * hz, np.asarray(e), np.asarray(al)

def t_reach(x, v, above=True):
    hit = x >= v if above else x <= v
    i = int(np.argmax(hit)); return i / HZ if hit[i] else float("nan")

if MODE == "stats":
    sh = SH
    print(f"parameters: g {sh.g}, C_Lmax {sh.cl_max} 1/m, a_max {sh.a_max} (8 g), cd0 {sh.cd0:.2e}, k_ind {sh.k_ind:.1f}, "
          f"engine {sh.power:.0f} W / {sh.f_static:.0f} N, coast climb {sh.a0}, climb/dive {sh.a_climb}/{sh.a_dive}, "
          f"store {sh.e_max / 1e3:.0f} kJ, soft p {2 ** sh.soft_k}, nose x{sh.nose_mult}, bullet {sh.bullet_cost:.1f} J")
    print(f"stall {math.sqrt(sh.g / sh.cl_max):.1f} m/s; corner {sh.u_corner:.1f} m/s; min radius {1 / sh.cl_max:.0f} m; max excess {PE:.0f} W")
    u, r, e, al = fly(sh, 105, 0, [0, 0, 0], 40); print(f"cruise (coast, charging) {np.mean(u[-HZ:]):.1f} m/s; store full after {t_reach(e, 0.98 * sh.e_max):.0f} s from empty")
    u, r, e, al = fly(sh, 105, sh.e_max, [0, 0, 0], 60); print(f"top level speed (store full) {np.mean(u[-HZ:]):.1f} m/s")
    u, r, e, al = fly(sh, 105, sh.e_max, [1, 0, 0], 10); print(f"dive from full store: peak {u.max():.0f} m/s, empty after {t_reach(e, 0.02 * sh.e_max, False):.1f} s")
    u, r, e, al = fly(sh, 105, 0, [2, 0, 0], 40)
    print(f"hold reverse from 105: {u[HZ]:.0f} m/s after 1 s; settles {np.mean(u[-HZ:]):.0f} m/s; store after 40 s {e[-1]:.0f} J")
    best = (0, 0)
    for u0 in (80, 85, 90, 95, 100, 105):
        u, r, e, al = fly(sh, u0, 5000, [0, 6, 0], 1.0)
        if np.abs(r).max() > best[0]: best = (np.abs(r).max(), int(np.argmax(np.abs(r) >= 0.9 * np.abs(r).max())) / HZ)
    print(f"instantaneous peak {math.degrees(best[0]):.0f} deg/s, nose t90 {best[1]:.2f} s")
    print("turn levels (coast, store full; datum = equilibrium speed held while turning):")
    for lvl, st in ((1 / 3, 2), (2 / 3, 4), (1.0, 6)):
        for u0 in (60.0, 110.0):
            u, r, e, al = fly(sh, u0, sh.e_max, [0, st, 0], 30)
            print(f"  {lvl:.2f} stick from {u0:.0f} m/s: {np.mean(u[-HZ:]):5.1f} m/s, {math.degrees(np.mean(r[-HZ:])):5.1f} deg/s, "
                  f"r {np.mean(u[-HZ:]) / np.mean(r[-HZ:]):4.0f} m, gun offset (alpha) {math.degrees(np.mean(al[-HZ:])):4.1f} deg")
    u, r, e, al = fly(dataclasses.replace(sh, e_max=1e9), 90, 5e8, [1, 6, 0], 25)
    print(f"sustained boost turn, unlimited store: {np.mean(u[-HZ:]):.0f} m/s, {math.degrees(np.mean(r[-HZ:])):.0f} deg/s")
    for name, act in (("coasting", [0, 0, 1]), ("climbing", [2, 0, 1]), ("diving", [1, 0, 1]), ("hard turn", [0, 6, 1])):
        u, r, e, al = fly(sh, 105, sh.e_max, act, 120)
        print(f"fire while {name:>9}: full store lasts {t_reach(e, sh.bullet_cost, False):.1f} s")
elif MODE == "acc":
    S = 1024
    ks = jax.random.split(jax.random.PRNGKey(0), 12)
    acts = jnp.stack([jax.random.randint(ks[0], (20, S), 0, 3), jax.random.randint(ks[1], (20, S), 0, 7), jax.random.randint(ks[11], (20, S), 0, 2)], -1)
    sp0 = jax.random.uniform(ks[2], (S,), minval=60.0, maxval=150.0); hd = jax.random.uniform(ks[3], (S,), minval=0.0, maxval=6.28)
    E0 = jax.random.uniform(ks[4], (S,), minval=500.0, maxval=9500.0)
    x0 = jax.random.uniform(ks[5], (S,), minval=-2000.0, maxval=2000.0); y0 = jax.random.uniform(ks[6], (S,), minval=-2000.0, maxval=2000.0)
    FLD = (jax.random.uniform(ks[7], (24,), minval=-2000.0, maxval=2000.0), jax.random.uniform(ks[8], (24,), minval=-2000.0, maxval=2000.0),
           jax.random.uniform(ks[9], (24,), minval=100.0, maxval=490.0), jnp.full((24,), 40.0), math.log(math.sqrt(2.0)) * jax.random.randint(ks[10], (24,), -2, 3).astype(jnp.float64))
    def run(sh, field, hz, dt):
        h = 1.0 / hz; tph = int(round(0.1 * hz))
        st = tuple(jnp.asarray(v, dt) for v in F.init(x0, y0, hd, sp0, E0, field)); fl = tuple(jnp.asarray(f, dt) for f in field)
        def hold(s, a):
            # one bullet per 30 Hz decision: fire only on the first sub-step of each decision
            def sub(i, s):
                aa = a.at[..., 2].set(jnp.where(i % (hz // 30) == 0, a[..., 2], 0))
                return F.step(sh, fl, s, aa, h)
            return jax.lax.fori_loop(0, tph, sub, s), None
        return jax.jit(lambda s: jax.lax.scan(hold, s, acts)[0])(st)
    for label, field in (("no fields", tuple(f[:0] for f in FLD)), ("fields", FLD)):
        ref = run(SH, field, 3840, jnp.float64)
        errs = [float(jnp.mean(jnp.hypot(o[0] - ref[0], o[1] - ref[1]))) for o in (run(SH, field, hz, jnp.float32) for hz in (30, 60, 120))]
        sh0 = dataclasses.replace(SH, cd0=0.0, k_ind=0.0, power=0.0, f_static=1e-6, bullet_cost=0.0)
        st0 = tuple(jnp.asarray(v, jnp.float32) for v in F.init(x0, y0, hd, sp0, E0, field))
        en = lambda s: 0.5 * s[4] ** 2 + s[9]
        st1 = run(sh0, field, 30, jnp.float32)
        print(f"[{label}] mean pos error 30/60/120 Hz {' / '.join(f'{e:.3f}' for e in errs)} m; energy drift (no drag/engine/fire) {float(jnp.mean(jnp.abs(en(st1) - en(st0)) / en(st0))):.1e}")
else:
    import today_jax as TJ
    big = 40960
    k = jax.random.split(jax.random.PRNGKey(1), 8)
    a = jnp.stack([jax.random.randint(k[0], (big,), 0, 3), jax.random.randint(k[1], (big,), 0, 7), jax.random.randint(k[7], (big,), 0, 2)], -1)
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
        print(f"final model, {label}: {bench(lambda s: F.step(SH, Fd, s, a, 1 / 30), F.init(xs, ys, 0 * xs + 1.0, 100.0 + 0 * xs, 5000.0 + 0 * xs, Fd)):.1f} ms")
