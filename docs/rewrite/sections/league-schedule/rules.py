import numpy as np
grid,mean=np.load("mean.npy")
E=lambda s: np.interp(s,grid,mean)
def show(name, pts):
    pts=[p for p in pts if p<=161]
    e=E(np.array(pts)); g=np.diff(e)
    print(f"{name:34s} n={len(pts):2d} gaps mean {g.mean():5.0f} sd {g.std():4.0f} min {g.min():4.0f} max {g.max():4.0f}")
    print("   ", [round(p,1) for p in pts]); print("   ", np.round(g).astype(int).tolist())
show("current", [5,10,15,20,25,30,40,50,60,75,90,110,130,160])
R10=[10,12.5,16,20,25,31.5,40,50,63,80,100,125,160]
show("R10 preferred numbers (x1.26)", R10)
for q in (1.2,1.22,1.25):
    show(f"geometric 10M x{q}", [10*q**k for k in range(20)])
# fitted: best power law s_k = a k^p and geometric start, for gap 100
from scipy.optimize import minimize
def loss(par, fam):
    a,p=par
    pts=[a*k**p for k in range(1,40)] if fam=="power" else [a*p**k for k in range(40)]
    pts=np.array([x for x in pts if x<=160])
    if len(pts)<3: return 1e9
    g=np.diff(E(pts)); return float(((g-100)**2).mean())
for fam,x0 in (("power",(2,1.5)),("geom",(8,1.2))):
    r=minimize(loss,x0,args=(fam,),method="Nelder-Mead"); print(fam, np.round(r.x,3), round(r.fun**.5,1))
