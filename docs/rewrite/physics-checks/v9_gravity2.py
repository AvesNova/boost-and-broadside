"""Ours with climb geometry (sin gamma = a/g, cap 60 deg) vs the 3D gravity point mass."""
exec(open(__import__("os").path.join(__import__("os").path.dirname(__file__), "v9_gravity.py")).read().split("cases = [")[0])
smax = math.sin(math.radians(60.0))
cases = [("coast (6 deg)", 0, sh.a0, 105.0, 5000.0), ("half climb (30 deg)", 2, g / 2, 105.0, 0.0),
         ("full climb, capped at 60 deg", 2, g * smax, 105.0, 0.0), ("full dive, capped at -60 deg", 1, -g * smax, 60.0, 90000.0)]
print("columns: speed m/s | energy J | horizontal distance m   (ref = 3D point mass with a pilot law; old = no geometry; geom = cos-gamma geometry)")
for name, pm, a_cmd, u0, E0 in cases:
    r = ref(u0, E0, a_cmd, 3.0)
    res = {}
    for label, geo in (("old", False), ("geom", True)):
        s_ = dataclasses.replace(sh, e_max=1e5, geom=geo, a_climb=abs(a_cmd) if pm == 2 else sh.a_climb, a_dive=abs(a_cmd) if pm == 1 else sh.a_dive)
        u, th, e, al = fly(s_, u0, E0, [pm, 0, 0], 3.0)
        st = F.init(jnp.array([0.0]), jnp.array([0.0]), jnp.array([0.0]), jnp.array([u0]), jnp.array([E0]), NOF)
        a = jnp.array([[pm, 0, 0]])
        xs = jax.jit(lambda s: jax.lax.scan(lambda s, i: (F.step(s_, NOF, s, a, 1 / HZ), F.step(s_, NOF, s, a, 1 / HZ)[0][0]), s, None, length=3 * HZ))(st)[1]
        res[label] = (u, e, np.asarray(xs))
    print(f"\n{name}")
    for t in (1.0, 2.0, 3.0):
        k = int(t / 0.5) - 1; i = int(t * HZ) - 1
        ru, re, rx, rg = r[k]
        o, gm = res["old"], res["geom"]
        print(f"  {t:.0f} s | ref {ru:5.1f} {re:7.0f} {rx:5.0f} | old {o[0][i]:5.1f} {o[1][i]:7.0f} {o[2][i]:5.0f} | geom {gm[0][i]:5.1f} {gm[1][i]:7.0f} {gm[2][i]:5.0f}")
