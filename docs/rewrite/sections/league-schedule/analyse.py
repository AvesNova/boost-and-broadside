import numpy as np
from sklearn.isotonic import IsotonicRegression
from data import rl, dm_wl, start
def curve(points, s0):
    s=np.array([0.0]+[p[0] for p in points]); e=np.array([s0]+[p[1] for p in points])
    w=np.array([1.0]+[1/p[2]**2 for p in points])
    return s, IsotonicRegression().fit(s,e,sample_weight=w).predict(s)
modes={"frontline":curve(rl["frontline"],start["frontline"]),
       "elimination":curve(rl["elimination"],start["elimination"]),
       "deathmatch (win/loss)":curve(dm_wl,start["deathmatch"])}
grid=np.linspace(0,160,1601)
interp={k:np.interp(grid,*v) for k,v in modes.items()}
mean=np.mean(list(interp.values()),0)
np.save("mean.npy",np.stack([grid,mean]))
for k,v in modes.items(): print(k, np.round(v[0],1).tolist(), np.round(v[1]).tolist())
for s in (0,5,10,15,20,25,30,40,50,60,75,90,110,130,160): print(s, round(float(np.interp(s,grid,mean))))
