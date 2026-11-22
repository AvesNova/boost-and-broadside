"""Held turns: each type x power x stick level x start (speed, store), 600 s, sampled every 1 s."""
import torch, math, json
from sim import run_batch, card
DEG = 180 / math.pi
rows = []
for t in range(3):
    for power in range(3):
        for stick in (2, 4, 6):
            for v0 in (145.0, 300.0):
                for e0 in (card.CARD["spawn_energy"][t], card.CARD["max_energy"][t]):
                    rows.append((t, v0, e0, stick, power))
T = torch.tensor
types, speeds, energies, sticks, powers = (T([r[i] for r in rows]) for i in range(5))
r = run_batch(types, speeds, energies, sticks, powers, 30 * 600, keep_every=30)
out = []
for i, (t, v0, e0, stick, power) in enumerate(rows):
    sp, en, rt = r["speed"][:, i], r["energy"][:, i], r["rate"][:, i] * DEG
    last = slice(-60, None)  # final 60 s
    slope = float(en[-1] - en[-61]) / 60.0
    rec = dict(type=t, v0=v0, e0=e0, stick=stick, power=power,
               speed=float(sp[last].mean()), speed_span=float(sp[last].max() - sp[last].min()),
               rate=float(rt[last].mean()), energy=float(en[-1]), energy_slope=slope,
               speed_series=sp.tolist(), energy_series=en.tolist(), rate_series=rt.tolist())
    rec["radius"] = rec["speed"] / (rec["rate"] / DEG)
    # first time the store reaches the 2 % fade band
    hit = (en < 0.02 * card.CARD["max_energy"][t]).nonzero()
    rec["store_empty_s"] = float(hit[0]) + 1 if len(hit) else None
    out.append(rec)
    print("%-6s pow %d stick %d v0 %3.0f E0 %3.0fk -> v %5.1f (span %4.1f) rate %5.1f R %6.1f E %6.1fk dE/dt %+7.1f J/s empty@%s" % (
        card.TYPE_NAMES[t], power, stick, v0, e0 / 1000, rec["speed"], rec["speed_span"], rec["rate"], rec["radius"],
        rec["energy"] / 1000, slope, rec["store_empty_s"]))
json.dump(out, open("sustained.json", "w"))
