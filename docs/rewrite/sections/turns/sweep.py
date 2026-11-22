"""Sustained-turn curve: continuous stick magnitude (scratch override of stick index 6), 600 s held."""
import torch, math, json
from sim import run_batch, card
DEG = 180 / math.pi
mags = [0.05 * k for k in range(1, 21)]
rows = []
for t in range(3):
    for power in range(3):
        for m in mags:
            for e0 in (card.CARD["spawn_energy"][t], card.CARD["max_energy"][t]):
                rows.append((t, 200.0, e0, 6, power, m))
T = torch.tensor
types, speeds, energies, sticks, powers, mag = (T([r[i] for r in rows]) for i in range(6))
r = run_batch(types, speeds, energies, sticks, powers, 30 * 600, keep_every=30, magnitude=mag.float())
out = []
for i, (t, v0, e0, stick, power, m) in enumerate(rows):
    sp, en, rt = r["speed"][:, i], r["energy"][:, i], r["rate"][:, i] * DEG
    rec = dict(type=t, e0=e0, power=power, magnitude=m, speed=float(sp[-60:].mean()), speed_span=float(sp[-60:].max() - sp[-60:].min()),
               rate=float(rt[-60:].mean()), energy=float(en[-1]), energy_slope=float(en[-1] - en[-61]) / 60)
    rec["radius"] = rec["speed"] / (rec["rate"] / DEG)
    out.append(rec)
    if e0 == card.CARD["spawn_energy"][t]:
        print("%-6s pow %d mag %.2f -> v %5.1f span %4.1f rate %5.1f R %6.0f E %6.1fk dE/dt %+6.1f" % (
            card.TYPE_NAMES[t], power, m, rec["speed"], rec["speed_span"], rec["rate"], rec["radius"], rec["energy"] / 1000, rec["energy_slope"]))
json.dump(out, open("sweep.json", "w"))
