"""Boost turns while the store lasts: full store, start 250 m/s, continuous stick, quasi-steady phase."""
import torch, math, json
from sim import run_batch, card
DEG = 180 / math.pi
import sys
V0 = float(sys.argv[1])
mags = [0.05 * k for k in range(6, 21)]
rows = [(t, V0, card.CARD["max_energy"][t], 6, 1, m) for t in range(3) for m in mags]
T = torch.tensor
types, speeds, energies, sticks, powers, mag = (T([r[i] for r in rows]) for i in range(6))
r = run_batch(types, speeds, energies, sticks, powers, 30 * 120, keep_every=15, magnitude=mag.float())
out = []
for i, (t, v0, e0, stick, power, m) in enumerate(rows):
    sp, en, rt = r["speed"][:, i], r["energy"][:, i], r["rate"][:, i] * DEG
    alive = (en > 0.05 * card.CARD["max_energy"][t]).nonzero()
    end = int(alive[-1]) if len(alive) else 0
    # quasi-steady: from 10 s (index 20) to the time the store reaches 5 %
    seg = slice(20, max(end, 21))
    rec = dict(type=t, magnitude=m, speed=float(sp[seg].mean()), rate=float(rt[seg].mean()),
               store_seconds=(end + 1) / 2, drain=float((en[20] - en[max(end, 21)]) / max((end - 20) / 2, 0.5)))
    rec["radius"] = rec["speed"] / (rec["rate"] / DEG)
    out.append(rec)
    print("%-6s mag %.2f  v %5.1f rate %5.1f R %5.0f  store lasts %5.1f s  drain %.1f kJ/s" % (
        card.TYPE_NAMES[t], m, rec["speed"], rec["rate"], rec["radius"], rec["store_seconds"], rec["drain"] / 1000))
json.dump(out, open(f"storefed_{int(V0)}.json", "w"))
