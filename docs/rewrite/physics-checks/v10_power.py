"""Engine power falling with the store: fill, fixed-store performance and accuracy (setup in v10_common)."""
from v10_common import *

print("\nFill from empty, coasting straight (time to 50 / 90 / 99% of the store; speed at the end):")
for n, sh in CURVES.items():
    u, E, _ = run(sh, [[0, 0, 0]] * 60, 104.0, 0.0, 30)
    t = [np.argmax(E >= f * sh.e_max) / 30 if (E >= f * sh.e_max).any() else float("nan") for f in (0.5, 0.9, 0.99)]
    print(f"- {n}: {t[0]:.1f} / {t[1]:.1f} / {t[2]:.1f} s; after 120 s store {E[-1] / sh.e_max:.3f}, speed {u[-1]:.1f} m/s")

print("\nAt a fixed store (no charging): straight top speed; coast datum at 0.75 and 1.0 stick:")
for n, sh in CURVES.items():
    rows = []
    for f in (0.0, 0.5, 0.75, 1.0):
        s0 = dataclasses.replace(sh, a0=0.0)
        top = tail(fly(s0, 110.0, f * sh.e_max, [0, 0, 0], 30)[0])
        u75, r75, _, _ = fly(s0, 60.0, f * sh.e_max, [0, 4, 0], 30)
        u1, r1, _, _ = fly(s0, 60.0, f * sh.e_max, [0, 6, 0], 30)
        rows.append(f"{f:.2f}: {top:.0f} m/s, {math.degrees(tail(r75)):.0f} / {math.degrees(tail(r1)):.0f} deg/s")
    print(f"- {n}: " + "; ".join(rows))

print("\nAccuracy crossing the knee (store 40% -> past 50%, mixed inputs, 12 s) vs a 960 Hz reference:")
acts = [[0, 0, 0], [0, 4, 0], [2, 0, 0], [0, 6, 0], [0, 0, 0], [1, 2, 0]]
for n, sh in CURVES.items():
    ru, rE, _ = run(sh, acts, 100.0, 0.4 * sh.e_max, 960)
    errs = []
    for hz in (30, 60, 120):
        u, E, _ = run(sh, acts, 100.0, 0.4 * sh.e_max, hz)
        k = 960 // hz
        errs.append((np.abs(u - ru[k - 1::k]).max(), np.abs(E - rE[k - 1::k]).max()))
    print(f"- {n}: max speed error " + " / ".join(f"{e[0]:.4f}" for e in errs) + " m/s; max store error "
          + " / ".join(f"{e[1]:.2f}" for e in errs) + " J (30 / 60 / 120 Hz)")
