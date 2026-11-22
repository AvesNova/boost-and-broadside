import torch, math
from sim import run, card
DEG = 180 / math.pi
for t in range(3):
    e0 = card.CARD["spawn_energy"][t]
    for power in range(3):
        for stick in (2, 4, 6):
            r = run(t, torch.tensor([145.0, 300.0]), torch.tensor([e0, e0]), stick, power, 30 * 600)
            for i in range(2):
                sp, en, rt = r["speed"][:, i], r["energy"][:, i], r["rate"][:, i] * DEG
                marks = [5, 30, 60, 120, 300, 600]
                print(card.TYPE_NAMES[t], "pow", power, "stick", stick, "v0", [145, 300][i],
                      " ".join("%ds:v%.0f,r%.1f,E%.0fk" % (m, sp[30*m-1], rt[30*m-1], en[30*m-1]/1000) for m in marks))
