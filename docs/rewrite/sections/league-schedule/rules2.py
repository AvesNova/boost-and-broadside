import numpy as np
from rules import show, E
for q in (1.15,1.16,1.17):
    show(f"geometric 10M x{q}", [10*q**k for k in range(30)])
R20=[10,11.2,12.5,14,16,18,20,22.4,25,28,31.5,35.5,40,45,50,56,63,71,80,90,100,112,125,140,160]
show("R20 (x1.122)", R20)
# gap per member in the stretch 15-90M only
for q in (1.15,1.16,1.17,1.2):
    p=np.array([10*q**k for k in range(30)]); p=p[(p>=15)&(p<=95)]
    print(q, "15-95M mean gap", round(float(np.diff(E(p)).mean())))
