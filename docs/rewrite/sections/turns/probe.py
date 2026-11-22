import torch, math
from sim import run, card
names = card.TYPE_NAMES
for t in range(3):
    for power in range(3):
        r = run(t, torch.tensor([145.0]), card.CARD["spawn_energy"][t], 0, power, 30*240)
        sp, en = r["speed"][:, 0], r["energy"][:, 0]
        print(names[t], "power", power, "straight: speed at 10,30,60,120,240 s",
              [round(float(sp[30*k-1]),1) for k in (10,30,60,120,240)],
              "energy", [round(float(en[30*k-1])/1000) for k in (10,30,60,120,240)])
