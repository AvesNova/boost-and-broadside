import numpy as np, math, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Circle
L=14
ax_=[(q,r) for q in range(-L,L+1) for r in range(-L,L+1)]
P=np.array([(q+r/2, r*math.sqrt(3)/2) for q,r in ax_]); d=np.hypot(*P.T)
ang=np.degrees(np.arctan2(P[:,1],P[:,0]))%60
hexG=np.array([max(abs(q),abs(r),abs(q+r))<=10 for q,r in ax_])
shell=np.abs(d-9.5394)<1e-3
pick=np.isclose(ang,5.2087,atol=1e-3)|np.isclose(ang,26.9955,atol=1e-3)|np.isclose(ang,33.0045,atol=1e-3)
disc=(d<=9.17)|(shell&pick)
assert hexG.sum()==331 and disc.sum()==331
Rplay=2600.0
panels=[("Today: 10-ring hexagon",hexG,10.0,"circle sticks out; 260 m cells"),
        ("Hexagon, 2σ buffer",hexG,7.68,"338 m cells"),
        ("Disc, 2σ buffer",disc,7.83,"332 m cells")]
IN="#4C78A8"; BUF="#C9D6E8"; EDGE="white"; RING="#D1495B"
fig,axs=plt.subplots(1,3,figsize=(15,5.6),facecolor="white")
for axp,(title,G,Rsp,sub) in zip(axs,panels):
    a=Rplay/Rsp
    nin=0
    for (x,y),g,dd in zip(P,G,d):
        if not g: continue
        inside=dd*a<=Rplay
        nin+=inside
        axp.add_patch(RegularPolygon((x*a,y*a),6,radius=a/math.sqrt(3),orientation=0,
                      facecolor=IN if inside else BUF,edgecolor=EDGE,lw=0.6))
    axp.add_patch(Circle((0,0),Rplay,fill=False,ec=RING,lw=2.2))
    lim=3600; axp.set_xlim(-lim,lim); axp.set_ylim(-lim,lim); axp.set_aspect("equal")
    axp.set_title(f"{title}\n{sub}, {nin} of 331 cells inside",fontsize=11)
    axp.set_xticks([-2600,0,2600]); axp.set_yticks([-2600,0,2600])
    axp.tick_params(labelsize=8); 
    for s in axp.spines.values(): s.set_visible(False)

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
fig.legend(handles=[Patch(fc=IN,label="cell centre inside playable circle"),Patch(fc=BUF,label="buffer cell"),
    Line2D([0],[0],color=RING,lw=2.2,label="playable boundary")],loc="lower center",ncol=3,frameon=False,fontsize=10)
fig.suptitle("Density grid: 331 cells over the playable area (metres at the 5v5 playable radius, 2600 m; HL-Gauss σ = 0.75 spacing)",fontsize=13)
fig.tight_layout(rect=(0,0.06,1,0.95))
fig.savefig("/home/user/boost-and-broadside/docs/rewrite/figures/density-grid-layouts.png",dpi=130)
