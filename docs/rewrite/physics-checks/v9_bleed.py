from v9_common import *
print("Full stick from 90 m/s, coast, store full (stall-angle thirds). Kinetic = 0.5 u^2 (J), total = kinetic + store.")
print(f"{'bleed from':>10} | {'speed 1 s / 3 s / 6 s':>22} | {'store 3 s / 6 s':>16} | {'total energy lost by 3 s':>24} | {'turn rate 1-3 s':>15} | {'datum after store empty':>24}")
for bleed in ("kinetic", "both", "store"):
    sh = ship(levels="stick", bleed=bleed)
    u, r, e, al = fly(sh, 90.0, sh.e_max, [0, 6, 0], 40)
    tot = 0.5 * u ** 2 + e
    tot0 = 0.5 * 90.0 ** 2 + sh.e_max
    print(f"{bleed:>10} | {u[HZ]:6.1f} / {u[3 * HZ]:5.1f} / {u[6 * HZ]:5.1f} m/s | {e[3 * HZ]:7.0f} / {e[6 * HZ]:5.0f} J | {tot0 - tot[3 * HZ]:20.0f} J | "
          f"{math.degrees(np.mean(r[HZ:3 * HZ])):11.1f} deg/s | {tail(u):6.1f} m/s, {math.degrees(tail(r)):5.1f} deg/s")
print("\n2/3 stick from 90 m/s, coast, store full:")
for bleed in ("kinetic", "both", "store"):
    sh = ship(levels="stick", bleed=bleed)
    u, r, e, al = fly(sh, 90.0, sh.e_max, [0, 4, 0], 40)
    i = int(np.argmax(e < 0.05 * sh.e_max)) if (e < 0.05 * sh.e_max).any() else -1
    print(f"{bleed:>10}: speed 3 s {u[3 * HZ]:5.1f}, 6 s {u[6 * HZ]:5.1f} m/s; store {e[3 * HZ]:5.0f} / {e[6 * HZ]:5.0f} J; "
          f"store empty after {i / HZ if i >= 0 else float('nan'):.1f} s; then {tail(u):.1f} m/s, {math.degrees(tail(r)):.1f} deg/s")
